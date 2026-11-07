"""STEP 3 of 4 - build and validate the run timeline.

    .venv\\Scripts\\python.exe scripts\\stages\\step3_timeline.py [--case LABEL]
                                                            [--no-validate]

Turns the profiling data step 1 wrote into output/run_timeline.html, the
interactive "post-game summary" of a run (x axis = wall time, one bar per
instrumented stage call, one row per worker lane).

This is also where the tenth validator runs.  validate_timeline.py belongs to
this step rather than to step 2 because it reads the HTML this step produces, and
running it before the HTML exists would report a failure that says nothing about
the model.

  * build  : scripts/build_run_timeline.py
             input  output/<label>/results/run_spans.csv + run_profile.csv
             output output/run_timeline.html
  * check  : scripts/validate_timeline.py
             layer [A] 8 checks on the payload and page structure
             layer [B] 27 checks that boot the page's own JS under Node

Layer [B] needs `node` on PATH.  Node is deliberately NOT a project dependency,
so when it is absent the suite reports [SKIP] - which is not a pass, and the
validator does not count it as one.  Nothing here treats that as a failure
either; the eight layer-[A] checks still ran.

--case LABEL restricts both the build and the validation to one case.
--no-validate builds the page and skips validate_timeline.py.

Runtime: ~20 s, dominated by the build.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stage_lib as L  # noqa: E402

HTML = os.path.join(L.OUT, "run_timeline.html")
TIMELINE_LOG = os.path.join(L.OUT, "validation", "run_timeline.txt")


def missing_spans(cases):
    """Configured cases that lack a run_spans.csv or run_profile.csv pair.

    build_run_timeline.py needs BOTH per case (it uses the aggregate profile for
    the function inventory), and it deliberately has no glob fallback: it only
    ever looks at the cases in site.cases.  Checking here means the failure is a
    clear sentence instead of a SystemExit from two files down.
    """
    out = []
    for case in cases:
        for name in ("run_spans.csv", "run_profile.csv"):
            p = os.path.join(L.OUT, case, "results", name)
            if not os.path.exists(p):
                out.append(f"output/{case}/results/{name}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", metavar="LABEL", default=None,
                    help="restrict the timeline to one case")
    ap.add_argument("--no-validate", action="store_true",
                    help="build the page but skip validate_timeline.py")
    args = ap.parse_args()

    L.banner(3, "RUN TIMELINE",
             "Builds output/run_timeline.html from the step-1 profiling data, "
             "then validates it.")

    cases = L.case_labels()
    if not cases:
        print("site.cases is empty - there is no per-case profiling data to "
              "build a timeline from.")
        L.log_stage_run("step3_timeline", 1, 0.0, "no site.cases")
        return 1

    check = list(cases)
    if args.case:
        if args.case not in cases:
            print(f"case {args.case!r} is not in site.cases (known: "
                  f"{', '.join(cases)})")
            return 1
        check = [args.case]

    missing = missing_spans(check)
    if missing:
        print(f"\nERROR: {len(missing)} profiling file(s) missing:")
        for m in missing[:12]:
            print(f"  - {m}")
        print("\nRun step 1 first - run_analysis.py writes run_spans.csv and "
              "run_profile.csv per case.")
        L.log_stage_run("step3_timeline", 1, 0.0,
                        f"{len(missing)} span/profile file(s) missing")
        return 1

    size0 = os.path.getsize(HTML) if os.path.exists(HTML) else None
    argv = ["--case", args.case] if args.case else []
    results = [L.run_step("build_timeline", "build_run_timeline.py", argv)]

    if results[0].status == L.FAIL:
        print("\nbuild failed - not validating a stale page.")
        L.log_stage_run("step3_timeline", 1, results[0].seconds, "build failed")
        return 1

    if not os.path.exists(HTML):
        print(f"\nERROR: {L.rel(HTML)} was not written.")
        L.log_stage_run("step3_timeline", 1, results[0].seconds, "no html written")
        return 1
    size1 = os.path.getsize(HTML)
    print(f"    {L.rel(HTML)} = {size1:,} bytes"
          + (f" (was {size0:,})" if size0 else ""))
    if size1 < 100_000:
        print("    WARNING: that is small for a page carrying six cases of "
              "span data; check which cases were included.")
    results[0].note = f"{size1:,} bytes"

    if not args.no_validate:
        results.append(L.run_step("validate_timeline", "validate_timeline.py",
                                  [], log=TIMELINE_LOG))

    print("\n" + L.table(results))
    counts = L.tally(results)
    print(f"\n{counts[L.PASS]} pass, {counts[L.FAIL]} fail, "
          f"{counts[L.WARN]} warn, {counts[L.SKIP]} skip")
    print("open: " + L.rel(HTML))
    print("note: validate_timeline layer [B] needs `node` on PATH and reports "
          "[SKIP] without it - that is not a pass.")

    rc = L.exit_code(results)
    L.log_stage_run("step3_timeline", rc, sum(r.seconds for r in results),
                    f"build {size1:,} B")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
