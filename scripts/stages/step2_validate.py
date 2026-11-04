"""STEP 2 of 4 - run the validators.

    .venv\\Scripts\\python.exe scripts\\stages\\step2_validate.py
                                        [--quick] [--only NAME,...]
                                        [--fail-fast] [--force]

Validates the results STEP 1 produced.  It runs no analysis of its own, so it
cannot change a result; it launches the project's existing validator scripts one
after another and reports what each one said.

Order is fast-to-slow, so a mistake surfaces in seconds instead of after the
two 4-7 minute benchmark sweeps:

    sanity_checks          assert-level physics/geometry self-checks
    validate_sghat         vs Ho 2011 / SGHAT v6 published values
    validate_norm_vs_code  BS EN 62471:2008 text vs the code, constant by constant
    validate_iec62471      IEC 62471 clauses, weights, decision tree
    validate_icnirp        ICNIRP 2013 cross-check  (reads the step-1 CSVs)
    validate_e903          ASTM E903-20-style reflectance + self-consistency
                           of the step-1 CSVs
    validate_cases         one reference paper = one case, six papers
    validate_charts        figure/result QA + parallel parity + detector FP/FN
    validate_kuantan       external benchmark: Kuantan (Sreenath 2020)
    validate_dupplin       external benchmark: Dupplin (ForgeSolar)

Two things about the flags are load-bearing and must not be "tidied up":

  * validate_charts is invoked as `--full --hit-test polygon`, which is the
    combination its 28 checks were last validated with;
  * validate_kuantan and validate_dupplin are invoked with NO --hit-test flag.
    Their default is "circular", which is what produced the committed logs
    (kuantan 5/5, dupplin 4/3).  Passing "polygon" would change their numbers
    and stop this stage from reproducing the reference.

validate_dupplin is reported as WARN, never FAIL: it sits at 4 passed / 3 failed
for a documented reason (a 37-vertex row polygon replaced by one solid
equivalent-area rectangle, and no obstruction model), not because of a defect.
That is a model limitation recorded in AGENTS.md, so it must not turn a whole
validation run red.

validate_timeline is NOT run here - it reads output/run_timeline.html, which
STEP 3 produces.  It is the eleventh validator and STEP 3 owns it.

--quick runs the first four only (~1 min) - enough to catch a broken import or
a changed constant.  --only takes short names (sanity, sghat, norm, iec,
icnirp, e903, cases, charts, kuantan, dupplin).  --fail-fast stops at the first
failure instead of running the rest.  --force downgrades a missing-input
precheck from an error to a warning, so the independent validators can still be
run without step 1.

Runtime: ~15 min for all ten; ~1 min with --quick.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stage_lib as L  # noqa: E402

VALIDATION = os.path.join(L.OUT, "validation")

# key, script, extra argv, log file under output/validation, warn_only
VALIDATORS = [
    ("sanity", "sanity_checks.py", [], None, False),
    ("sghat", "validate_sghat.py", [], None, False),
    ("norm", "validate_norm_vs_code.py", [], "norm_vs_code.txt", False),
    ("iec", "validate_iec62471.py", [], "iec62471.txt", False),
    ("icnirp", "validate_icnirp.py", [], "icnirp.txt", False),
    ("e903", "validate_e903.py", [], "e903.txt", False),
    ("cases", "validate_cases.py", [], None, False),
    ("charts", "validate_charts.py", ["--full", "--hit-test", "polygon"],
     "charts.txt", False),
    ("kuantan", "validate_kuantan.py", [], "kuantan.txt", False),
    ("dupplin", "validate_dupplin.py", [], "dupplin.txt", True),
]
QUICK = {"sanity", "sghat", "norm", "iec"}


def needs_step1(keys):
    """Which selected validators read the CSVs step 1 writes."""
    return {"icnirp", "e903"} & set(keys)


def step1_ready():
    """Check that step 1 has actually run, and that what step 2 reads is there.

    Returns (missing_summary, missing_detail):

      * missing_summary - a configured case has no results/summary.csv, i.e.
        step 1 has not been run for it.  This is a blocker.
      * missing_detail - one of the nine COMMITTED case/receptor pairs has no
        glare_hours_*.csv.  validate_icnirp [I] evaluates all nine and fails if
        any is missing, so this is a blocker too.

    A case with NO glare at all legitimately has no per-pair detail file -
    run_analysis.py only writes those for non-empty details - which is why this
    checks the nine committed pairs, each against its OWN case, rather than
    "every case has a detail file".  The pair is not the cross product: a
    receptor that only glares in the az278 cases has no detail file in the
    az098 ones.  az278_e903_t20 has no committed pair at all, because all 65 of
    its receptors are zero.

    The file is matched as glare_hours_*__<receptor>.csv rather than by the
    hard-coded "PV-A__" prefix the two validators use, so a config with more
    than one array still resolves.
    """
    import fnmatch
    missing_summary, missing_detail = [], []
    for case in L.case_labels():
        if not os.path.exists(os.path.join(L.OUT, case, "results",
                                           "summary.csv")):
            missing_summary.append(case)
    for label, receptor, *_ in L.COMMITTED:
        res = os.path.join(L.OUT, label, "results")
        if not os.path.isdir(res) or not any(
                fnmatch.fnmatch(f, f"glare_hours_*__{receptor}.csv")
                for f in os.listdir(res)):
            missing_detail.append(f"{label}/{receptor}")
    return missing_summary, missing_detail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="run only the four fast validators (~1 min)")
    ap.add_argument("--only", metavar="NAME,...", default=None,
                    help="comma-separated subset of: "
                         + ", ".join(k for k, *_ in VALIDATORS))
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop at the first failure")
    ap.add_argument("--force", action="store_true",
                    help="warn instead of stopping when step-1 output is missing")
    args = ap.parse_args()

    L.banner(2, "VALIDATE",
             "Validates what step 1 produced.  Runs no analysis, changes no "
             "result.")

    if args.quick and args.only:
        print("--quick and --only are mutually exclusive; --only wins.")
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - {k for k, *_ in VALIDATORS}
        if unknown:
            print(f"unknown validator(s): {', '.join(sorted(unknown))}")
            print(f"known: {', '.join(k for k, *_ in VALIDATORS)}")
            return 1
        selected = [v for v in VALIDATORS if v[0] in wanted]
    elif args.quick:
        selected = [v for v in VALIDATORS if v[0] in QUICK]
    else:
        selected = list(VALIDATORS)

    keys = [v[0] for v in selected]
    print(f"running {len(selected)} of {len(VALIDATORS)} validators: "
          f"{', '.join(keys)}")
    if "dupplin" in keys:
        print("note: validate_dupplin is a documented model limitation "
              "(4/3) and is reported as WARN, not FAIL.")

    # ---- precheck: the two validators that read step-1 output ---------------
    readers = sorted(needs_step1(keys))
    not_ready = []
    if readers:
        missing_summary, missing_detail = step1_ready()
        not_ready = missing_summary + missing_detail
        if not_ready:
            detail = (f"step 1 output incomplete for {len(not_ready)} "
                      f"item(s) but {'/'.join(readers)} read it")
            if args.force:
                print(f"\nWARNING: {detail}; --force -> those become SKIP")
            else:
                print(f"\nERROR: {detail}")
                for item in not_ready[:8]:
                    print(f"  - output/{item}")
                if len(not_ready) > 8:
                    print(f"  ... and {len(not_ready) - 8} more")
                print("\nRun step 1 first, or pass --force to skip them.")
                L.log_stage_run("step2_validate", 1, 0.0,
                                f"precheck: {len(not_ready)} step-1 item(s) missing")
                return 1

    results = []
    for key, script, argv, log, warn_only in selected:
        path = os.path.join(VALIDATION, log) if log else None
        if key in readers and not_ready:
            results.append(L.StepResult(
                key, [script, *argv], 0, 0.0, L.SKIP,
                f"step-1 output incomplete ({len(not_ready)} item(s)"))
            print(f"--- {script}: SKIP (run step 1 first)")
            continue
        r = L.run_step(key, script, argv, warn_only=warn_only, log=path)
        results.append(r)
        if r.status == L.FAIL and args.fail_fast:
            print("--fail-fast: stopping here.")
            break

    print("\n" + L.table(results))
    counts = L.tally(results)
    print(f"\n{counts[L.PASS]} pass, {counts[L.FAIL]} fail, "
          f"{counts[L.WARN]} warn, {counts[L.SKIP]} skip")
    print("logs: " + L.rel(VALIDATION) + "/  (most validators APPEND, so the "
          "size deltas above are expected to grow)")

    rc = L.exit_code(results)
    note = f"{counts[L.PASS]}/{len(results)} pass" + (
        f", {counts[L.WARN]} warn" if counts[L.WARN] else "")
    L.log_stage_run("step2_validate", rc, sum(r.seconds for r in results), note)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
