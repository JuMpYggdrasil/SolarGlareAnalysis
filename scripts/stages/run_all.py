"""Run the three workflow stages in order: 1 analysis -> 2 validate -> 3 timeline.

    .venv\\Scripts\\python.exe scripts\\stages\\run_all.py
    .venv\\Scripts\\python.exe scripts\\stages\\run_all.py --dry-run
    .venv\\Scripts\\python.exe scripts\\stages\\run_all.py --from 3
    .venv\\Scripts\\python.exe scripts\\stages\\run_all.py --to 2 --keep-going
    .venv\\Scripts\\python.exe scripts\\stages\\run_all.py --stage1-args "--case az098_e903_t10"

Each stage is a separate script and can always be run on its own; this is only a
driver that runs them back to back, so "analysis, then validate, then timeline"
is one command instead of three.  It adds no behaviour of its own -
each stage is launched exactly as it would be if typed by hand, and the stage's
own exit code is the stage's own verdict.

The stages are ordered, not independent, and the dependency is the only reason
the order matters:

    1 writes output/<label>/** and is what step 2 validates and step 3 turns
      into a timeline;
    2 reads those results (e903 [D], icnirp [I]);
    3 reads the per-case run_spans.csv / run_profile.csv from step 1 and produces
      the HTML that validate_timeline.py then checks.

With --keep-going the chain continues past a failing stage.  Without it the
chain stops at the first failure, because stage 2 talking about numbers that
stage 1 just changed is not useful.

--dry-run prints the plan, checks each stage's prerequisites and writes
nothing and runs nothing, so the plan can be checked without spending 12
minutes.

Total: ~12 min for 1 case (1 ≈ 2 min, 2 ≈ 10 min, 3 ≈ 20 s); ~28 min for 6 cases.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stage_lib as L  # noqa: E402

STAGES = [
    (1, "step1_analysis.py", "analysis", []),
    (2, "step2_validate.py", "validate all ten validators", []),
    (3, "step3_timeline.py", "build + validate run_timeline.html", []),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def prereq(number):
    """(ok, message) for a stage's inputs, checked without running anything."""
    cases = L.case_labels()
    if number == 1:
        if not os.path.exists(L.CONFIG):
            return False, f"{L.rel(L.CONFIG)} not found"
        return True, f"{len(cases)} case(s) in site.cases" if cases \
            else "site.cases is empty"
    if number == 2:
        if not cases:
            return True, "no site.cases to check"
        missing = [f"{case}/results/summary.csv" for case in cases
                   if not os.path.exists(os.path.join(
                       L.OUT, case, "results", "summary.csv"))]
        # A zero-glare case has no per-pair detail file by design, so only the
        # committed pairs are required, each against its own case.
        for label, receptor, *_ in L.COMMITTED:
            res = os.path.join(L.OUT, label, "results")
            if not os.path.isdir(res) or not any(
                    f.endswith(f"__{receptor}.csv") for f in os.listdir(res)):
                missing.append(f"{label}/{receptor} detail")
        if missing:
            return False, (f"step 1 output incomplete: {len(missing)} item(s), "
                           f"e.g. {', '.join(missing[:2])}")
        return True, f"summary.csv + 9 committed pairs for {len(cases)} case(s)"
    if number == 3:
        if not cases:
            return False, "no site.cases -> no per-case profiling data"
        missing = [f"{c}/results/{n}" for c in cases
                   for n in ("run_spans.csv", "run_profile.csv")
                   if not os.path.exists(os.path.join(L.OUT, c, "results", n))]
        if missing:
            return False, f"missing {len(missing)} profiling file(s)"
        return True, f"run_spans.csv + run_profile.csv for {len(cases)} case(s)"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="first", type=int, choices=(1, 2, 3),
                    default=1, help="first stage to run (default 1)")
    ap.add_argument("--to", dest="last", type=int, choices=(1, 2, 3),
                    default=3, help="last stage to run (default 3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the prerequisite state, run nothing")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue past a failing stage")
    ap.add_argument("--stage1-args", metavar="ARGS", default=None,
                    help='extra argv for stage 1, e.g. "--case az098_e903_t10"')
    args = ap.parse_args()

    if args.first > args.last:
        print(f"--from {args.first} is after --to {args.last}")
        return 2

    plan = [s for s in STAGES if args.first <= s[0] <= args.last]
    L.banner("ALL", f"RUNNING STAGES {args.first} -> {args.last}",
             " ".join(f"{n}={name}" for n, name, _, _ in plan))

    print(f"{'stage':<6}{'script':<24}{'what':<36}prerequisite")
    print("-" * 100)
    blocked = set()
    for number, script, what, _ in plan:
        ok, msg = prereq(number)
        if not ok:
            blocked.add(number)
        print(f"{number:<6}{script:<24}{what:<36}"
              f"{'OK  ' if ok else 'MISS'} {msg}")
    print("-" * 100)

    est = {1: "~2 min (1 case)", 2: "~10 min", 3: "~20 s"}
    total = " ".join(f"{n}:{est[n]}" for n, *_ in plan)
    print(f"estimated: {total}")

    if args.dry_run:
        print("\n--dry-run: nothing was run and nothing was written.")
        if blocked:
            print("blocked stage(s): " + ", ".join(str(b) for b in sorted(blocked)))
            return 1
        return 0

    if blocked:
        # A blocked prerequisite is worth reporting even without --dry-run: the
        # first blocked stage is the one whose inputs are missing, and running
        # the rest would produce partial output that looks complete.
        first_blocked = min(blocked)
        if first_blocked == 1:
            print("\nERROR: stage 1 cannot start; nothing to do.")
            L.log_stage_run("run_all", 1, 0.0, "stage 1 blocked")
            return 1

    results = []
    t_start = time.time()
    for number, script, what, argv in plan:
        ok, msg = prereq(number)
        if not ok and number != 1:
            print(f"\n!!! stage {number} prerequisite not met: {msg}")
            print(f"!!! skipping it (and everything after it in the chain).")
            results.append(L.StepResult(f"stage{number}", [script], 1, 0.0,
                                        L.FAIL, msg))
            break
        extra = []
        if number == 1 and args.stage1_args:
            extra = args.stage1_args.split()
        r = L.run_step(f"stage{number}", script, argv + extra)
        if not r.note:
            r.note = what
        results.append(r)
        if r.status == L.FAIL and not args.keep_going:
            print(f"\nstage {number} failed (rc={r.rc}); stopping the chain.")
            print("Use --keep-going to run the remaining stages anyway.")
            break

    print("\n" + L.table(results, width=12))
    counts = L.tally(results)
    dt = time.time() - t_start
    print(f"\n{counts[L.PASS]} pass, {counts[L.FAIL]} fail, "
          f"{counts[L.WARN]} warn, {counts[L.SKIP]} skip   "
          f"chain wall time {dt / 60:.1f} min")
    rc = L.exit_code(results)
    print("artefacts: output/<label>/  output/run_timeline.html")
    L.log_stage_run("run_all", rc, dt,
                    f"stages {args.first}-{args.last}: "
                    f"{counts[L.PASS]} pass {counts[L.FAIL]} fail "
                    f"{counts[L.WARN]} warn")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
