"""Run the CoupleVibe project builder on a task and report.

  python -m eval.run_builder "<task>" [outdir]

Decompose (v12, grammar-locked JSON) -> generate each file (Reasoner) -> write. Prints the
plan, per-file line counts, and a PASS/FAIL on whether every file is real (non-stub).
"""
import sys

from agent.builder import build


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else (
        "Build a single-page To-Do web app: index.html (structure), style.css (clean modern "
        "styling), app.js (add/delete tasks, mark complete with strikethrough, persist in "
        "localStorage). Link the css and js from the html.")
    outdir = sys.argv[2] if len(sys.argv) > 2 else "scratch_build"
    rep = build(task, outdir)
    print("\n=== REPORT ===")
    print("files:", rep["steps"])
    print("real :", rep["written"])
    print("stubs:", rep["stubs"])
    print("RESULT:", "PASS (all files real)" if rep["ok"] else "FAIL (stubs present)")
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
