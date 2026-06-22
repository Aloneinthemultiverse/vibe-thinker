"""Generate a CPU Kaggle kernel that builds a ~15k-example tool-calling SFT set.

Why a separate CPU kernel: local HF access is blocked here, and reformatting public
data is parser-heavy. A CPU kernel costs no GPU quota and iterates fast, so we get the
15k dataset RIGHT before spending GPU on training it.

Sources merged into our ChatML action protocol ({"tool": name, "args": {...}}):
  - Glaive function-calling v2 (open, ~113k multi-turn) -> teaches general tool-use over
    hundreds of different tool-sets (anti-overfit: variable tool catalogue per example).
  - our 49 verified in-domain bug-fix examples (embedded) -> keeps our specific tools.

Output: /kaggle/working/sft_15k.jsonl  (+ a diversity/validation report).
Run:  python -m train.make_data_kernel  then  kaggle kernels push -p data_kernel
"""
import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "data_kernel")
USER = "sujitnarrayan"
SLUG = "vibethinker-build-15k"
TARGET = 15000

# In-domain supervision embedded into the kernel = our 49 bug-fix examples PLUS the
# decomposition (planner) examples, so v12 learns BOTH tool-use and task-splitting.
_ours = b""
for _f in ("sft_all.jsonl", "sft_plans.jsonl"):
    _p = os.path.join(ROOT, "data", _f)
    if os.path.exists(_p):
        with open(_p, "rb") as f:
            _ours += f.read()
            if not _ours.endswith(b"\n"):
                _ours += b"\n"
OURS_B64 = base64.b64encode(_ours).decode()

KERNEL = r'''
# Build ~15k tool-calling SFT examples (Glaive -> our action protocol + our in-domain).
# CPU kernel. Auto-generated.
import ast, base64, json, re, subprocess, sys, os

TARGET = __TARGET__

def loose(s):
    "Parse JSON OR a Python-literal dict (Glaive wraps args in single quotes)."
    try: return json.loads(s)
    except Exception: return ast.literal_eval(s)
subprocess.run("pip -q install -U datasets", shell=True)
from datasets import load_dataset

os.makedirs("/kaggle/working", exist_ok=True)

# --- our constitution preamble (generic; per-example tool list appended) ----------
PREAMBLE = (
"You are the reasoning core of an autonomous coding agent (constitution v1.0.0).\n\n"
"ROLE: You PROPOSE one action at a time. You NEVER execute anything yourself — a "
"deterministic orchestrator runs your proposed action and returns the result.\n\n"
"OUTPUT FORMAT (every turn): reason briefly, then emit EXACTLY ONE action as a JSON "
"object, e.g. {\"tool\": \"name\", \"args\": {...}}.\n\n"
"DISCIPLINE:\n- One action per turn. Use only the tools listed below.\n"
"- Do not repeat an identical action; if blocked, change your approach.\n\n"
"SAFETY: dangerous actions require human approval and are denied by default.")

def brace_objects(s):
    "Yield top-level {...} JSON substrings (brace-balanced, string-aware)."
    depth=0; start=None; instr=False; esc=False
    for i,ch in enumerate(s):
        if instr:
            if esc: esc=False
            elif ch=="\\": esc=True
            elif ch=='"': instr=False
            continue
        if ch=='"': instr=True
        elif ch=="{":
            if depth==0: start=i
            depth+=1
        elif ch=="}":
            if depth>0:
                depth-=1
                if depth==0 and start is not None: yield s[start:i+1]

def parse_tools(system):
    "Glaive system holds 1+ function-def JSON objects after a preamble line."
    defs=[]
    for raw in brace_objects(system):
        try:
            d=json.loads(raw)
            if isinstance(d,dict) and "name" in d: defs.append(d)
        except Exception: pass
    return defs

def render_tools(defs):
    lines=["Available tools:"]
    for d in defs:
        params=(d.get("parameters") or {}).get("properties") or {}
        keys=",".join(params.keys()) if isinstance(params,dict) else ""
        lines.append(f'- {d["name"]}  args: {{{keys}}}  -> {d.get("description","")[:80]}')
    return "\n".join(lines)

_ACT = re.compile(r"<functioncall>\s*(\{.*?\})", re.DOTALL)

def convert_glaive(system, chat):
    "Return our messages[] or None if it has no usable tool call."
    defs=parse_tools(system)
    if not defs: return None
    sys_msg=PREAMBLE+"\n\n"+render_tools(defs)
    msgs=[{"role":"system","content":sys_msg}]
    # split chat into role segments
    parts=re.split(r"\n*(USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*", chat)
    # parts[0] junk, then (marker, text) pairs
    it=iter(parts[1:]); had_call=False
    for marker in it:
        text=next(it,"").replace("<|endoftext|>","").strip()
        if not text: continue
        if marker=="USER:":
            msgs.append({"role":"user","content":text})
        elif marker=="FUNCTION RESPONSE:":
            msgs.append({"role":"user","content":"TOOL RESULT: "+text})
        elif marker=="ASSISTANT:":
            if "<functioncall>" in text:
                after=text.split("<functioncall>",1)[1]
                obj=next(brace_objects(after), None)   # brace-balanced, handles nested args
                if not obj: continue
                try:
                    call=loose(obj)
                    args=call.get("arguments") or {}
                    if isinstance(args,str):
                        args=loose(args) if args.strip().startswith("{") else {"input":args}
                    if not isinstance(args,dict): args={}
                    action=json.dumps({"tool":call["name"],"args":args},separators=(",",": "))
                    msgs.append({"role":"assistant",
                                 "content":"Call the tool to get what's needed.\n"+action})
                    had_call=True
                except Exception:
                    continue   # skip just this turn, not the whole example
            else:
                # plain final answer -> SKIP. We used to emit a {"tool":"respond"} action
                # here, but our agent loop has NO respond tool, so v11 learned to narrate
                # via respond and stalled on multi-step tasks. Dropping these turns keeps
                # only real tool-call supervision. (v11->v12 fix 2026-06-22)
                continue
    if not had_call: return None
    if msgs[-1]["role"]!="assistant": return None
    return msgs

# --- build ----------------------------------------------------------------------
print("loading Glaive v2 (streaming)...", flush=True)
ds=load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
pub=[]; seen=set(); scanned=0; dropped=0
for row in ds:
    scanned+=1
    msgs=convert_glaive(row.get("system",""), row.get("chat",""))
    if not msgs: dropped+=1; continue
    k=json.dumps(msgs,sort_keys=True)
    if k in seen: continue
    seen.add(k); pub.append({"messages":msgs})
    if len(pub)>=TARGET: break
    if scanned%5000==0: print(f"  scanned {scanned}, kept {len(pub)}", flush=True)

# our in-domain
ours=[json.loads(l) for l in base64.b64decode("__OURS_B64__").decode().splitlines() if l.strip()]
allrows=ours+pub  # in-domain first
with open("/kaggle/working/sft_15k.jsonl","w",encoding="utf-8") as f:
    for r in allrows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

# report
tools=set()
for r in allrows:
    for m in r["messages"]:
        if m["role"]=="assistant":
            mt=re.search(r'"tool":\s*"(\w+)"',m["content"])
            if mt: tools.add(mt.group(1))
print(f"\n==== sft_15k.jsonl ====")
print(f"scanned {scanned} glaive rows | kept public {len(pub)} | dropped {dropped} | ours {len(ours)}")
print(f"TOTAL {len(allrows)} examples | {len(tools)} DISTINCT tools")
print("sample tools:", sorted(tools)[:25])
print("\n--- sample public example (first 3 turns) ---")
for m in pub[0]["messages"][:3]:
    print(f"[{m['role']}] {m['content'][:140]}")
'''.replace("__TARGET__", str(TARGET)).replace("__OURS_B64__", OURS_B64)

META = {
    "id": f"{USER}/{SLUG}", "title": SLUG, "code_file": "kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True,
    "enable_gpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
}

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "kernel.py"), "w", encoding="utf-8") as f:
    f.write(KERNEL)
with open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(META, f, indent=2)
print(f"wrote {OUT}/kernel.py and kernel-metadata.json -> {META['id']} (CPU, target {TARGET})")
