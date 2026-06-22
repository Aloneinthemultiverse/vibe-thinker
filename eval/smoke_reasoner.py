"""Phase C gate (§9.7): prove the Reasoner server on :8082 is the BASE model and
actually reasons LONG — not v12 (whose 'reason briefly' training we are routing around).

Run: python -m eval.smoke_reasoner
Exit 0 = gate passes (safe to wire the duo loop). Exit 1 = do NOT wire; wrong model
or brevity training present.
"""
import re
import sys

from agent import llm

HARD = ("A function should return the running median after each insert into a stream of "
        "integers. Think step by step about the data structures and edge cases BEFORE "
        "writing any code. Reason thoroughly.")

MIN_CHARS = 800                      # base CoT should be long
STEP_MARKERS = re.compile(r"\bstep\b|\bfirst\b|\bthen\b|\bbecause\b|\bconsider\b", re.I)


def main():
    if not llm.reasoner_healthy():
        print(f"GATE FAIL: reasoner server not reachable at {llm.REASONER_ENDPOINT}")
        print("  start it:  llama-server -m <BASE-VibeThinker-3B>.gguf --port 8082 -ngl 0")
        return 1
    msg = [{"role": "system", "content": "You are a careful reasoning assistant."},
           {"role": "user", "content": HARD}]
    reply = llm.chat_reasoner(msg, temperature=0.6, max_tokens=4096)
    n = len(reply)
    markers = len(STEP_MARKERS.findall(reply))
    long_enough = n >= MIN_CHARS
    reasons = markers >= 3
    print(f"reply chars={n}  step-markers={markers}")
    print(f"  long_enough(>= {MIN_CHARS}): {long_enough}")
    print(f"  reasons(>=3 markers):       {reasons}")
    if long_enough and reasons:
        print("\nGATE PASS: reasoner produces long CoT — safe to wire duo loop.")
        return 0
    print("\nGATE FAIL: reply too short / not reasoning. Likely v12 on :8082, not base.")
    print("--- first 400 chars ---")
    print(reply[:400])
    return 1


if __name__ == "__main__":
    sys.exit(main())
