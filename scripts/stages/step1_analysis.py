"""STEP 1 of 4 - run the analysis.

    .venv\\Scripts\\python.exe scripts\\stages\\step1_analysis.py [--case LABEL]
                                                             [--no-verify]

Runs scripts/run_analysis.py (once per entry of site.cases, writing to
output/<label>/) and then checks the numbers against the committed results.

This step computes NOTHING itself.  It launches the analysis script, then reads
back output/<label>/results/*.csv and compares them with the reference table in
_stage_lib.COMMITTED, so "the results are the same as before" is a machine
check on every run rather than something a reader has to take on faith:

  * G / Y / R, minutes_with_glare and iec_worst_risk must match EXACTLY;
  * annual_peak_retinal_wcm2 must match within 1e-3 relative;
  * the per-case G/Y/R sums must match, which also proves that no receptor
    outside the nine committed rows started reporting glare;
  * the six-case sweep must sum to G 229 / Y 2543 / R 0.

Exit code is 0 only when every committed row matches.  A mismatch is reported
as a diff and nothing is repaired - a changed result is a finding, not
something to paper over.

--case LABEL passes straight through to run_analysis.py, which then runs that
single case (and writes only that case's output/<label>/); --no-verify skips the
comparison, which is what you want for a one-off case that is not part of the
committed sweep.

Runtime: ~110 s per case, ~11 min for the six committed cases.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stage_lib as L  # noqa: E402


def verify(cases):
    """Compare the committed reference against the CSVs just written.

    Returns (ok, lines).  `cases` is the list of case labels to check; it is the
    configured sweep, or the single label passed with --case.
    """
    lines = []
    ok = True
    index = {(c, r): (g, y, rd, m, rg, pk)
             for c, r, g, y, rd, m, rg, pk in L.COMMITTED}
    checked_cases = []

    for case in cases:
        rows, path = L.read_summary(case)
        if rows is None:
            ok = False
            lines.append(f"  MISSING  {L.rel(path)}")
            continue
        checked_cases.append(case)
        by_receptor = {r["receptor"]: r for r in rows}

        for (c, receptor), exp in index.items():
            if c != case:
                continue
            g, y, rd, m, rg, pk = exp
            row = by_receptor.get(receptor)
            if row is None:
                ok = False
                lines.append(f"  MISSING  {case} / {receptor} (not in summary.csv)")
                continue
            got = (int(row["G"]), int(row["Y"]), int(row["R"]),
                   int(row["minutes_with_glare"]), row["iec_worst_risk"])
            got_pk = float(row["annual_peak_retinal_wcm2"])
            pk_ok = abs(got_pk - pk) <= L.PEAK_RTOL * abs(pk)
            ints_ok = got == (g, y, rd, m, rg)
            ok = ok and ints_ok and pk_ok
            flag = "OK  " if ints_ok and pk_ok else "FAIL"
            lines.append(f"  {flag}  {case:<15} {receptor:<16} "
                         f"G/Y/R {got[0]}/{got[1]}/{got[2]}  "
                         f"min {got[3]:<4} {got[4]:<3} "
                         f"peak {got_pk:.4f} (exp {pk:.4f})")
            if not ints_ok:
                lines.append(f"        expected G/Y/R {g}/{y}/{rd}  "
                             f"min {m}  {rg}")
            if not pk_ok:
                lines.append(f"        expected peak {pk:.6f} "
                             f"({abs(got_pk - pk) / abs(pk):.2%} off)")

        want = L.EXPECTED_TOTALS.get(case)
        if want is not None:
            got_tot = L.case_totals(rows)
            tot_ok = got_tot == want
            ok = ok and tot_ok
            n_hot = len(L.glare_rows(rows))
            lines.append(f"  {'OK  ' if tot_ok else 'FAIL'}  {case:<15} "
                         f"case totals G/Y/R {got_tot[0]}/{got_tot[1]}/{got_tot[2]} "
                         f"(exp {want[0]}/{want[1]}/{want[2]})   "
                         f"{n_hot} of {len(rows)} receptors with glare")
            if not tot_ok:
                lines.append(f"        expected case totals {want}")

    # Sweep total, only meaningful when the whole committed sweep was run.
    if set(checked_cases) >= set(L.EXPECTED_CASES):
        tot = [0, 0, 0]
        for case in checked_cases:
            rows, _ = L.read_summary(case)
            for i, v in enumerate(L.case_totals(rows)):
                tot[i] += v
        got = tuple(tot)
        sweep_ok = got == L.SWEEP_TOTALS
        ok = ok and sweep_ok
        lines.append(f"  {'OK  ' if sweep_ok else 'FAIL'}  {'SWEEP TOTAL':<15} "
                     f"G/Y/R {got[0]}/{got[1]}/{got[2]} "
                     f"(exp {L.SWEEP_TOTALS[0]}/{L.SWEEP_TOTALS[1]}/"
                     f"{L.SWEEP_TOTALS[2]})")
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", metavar="LABEL", default=None,
                    help="run a single case from site.cases only")
    ap.add_argument("--no-verify", action="store_true",
                    help="do not compare the results with the committed reference")
    args = ap.parse_args()

    L.banner(1, "ANALYSIS",
             "Full-year per-minute analysis, one run per site.cases entry, "
             "plus every report figure.")

    if not os.path.exists(L.CONFIG):
        print(f"config not found: {L.rel(L.CONFIG)}")
        L.log_stage_run("step1_analysis", 1, 0.0, "config missing")
        return 1

    cases = L.case_labels()
    if not cases:
        print("site.cases is empty - run_analysis.py will write straight to "
              "output/ and the committed check does not apply.")
        targets = []
    elif args.case:
        if args.case not in cases:
            print(f"case {args.case!r} is not in site.cases (known: "
                  f"{', '.join(cases)})")
            L.log_stage_run("step1_analysis", 1, 0.0, "unknown case")
            return 1
        targets = [args.case]
    else:
        targets = cases

    argv = ["--brief"]
    if args.case:
        argv += ["--case", args.case]
    print(f"running: run_analysis.py {' '.join(argv)}")
    print(f"cases  : {', '.join(targets) if targets else '(none configured)'}")
    print("(--brief trims the console only - CSVs, figures and profile rows "
          "are identical)\n")
    rc, dt = L.run_script("run_analysis.py", argv)

    if rc != 0:
        print(f"\nrun_analysis.py failed (rc={rc}) - nothing else to do.")
        L.log_stage_run("step1_analysis", rc, dt, "run_analysis failed")
        return rc

    # Artefacts each case must have produced, so a later stage cannot discover
    # a missing file halfway through.
    missing = []
    for case in targets:
        for rel_path in (f"output/{case}/results/summary.csv",
                         f"output/{case}/results/run_profile.csv",
                         f"output/{case}/results/run_spans.csv"):
            if not os.path.exists(os.path.join(L.ROOT, rel_path)):
                missing.append(rel_path)
    if len(targets) > 1:
        rollup = "output/run_profile_all_cases.csv"
        if not os.path.exists(os.path.join(L.ROOT, rollup)):
            missing.append(rollup)
    if missing:
        print("\nMISSING OUTPUT after a successful run:")
        for m in sorted(set(missing)):
            print(f"  - {m}")
        L.log_stage_run("step1_analysis", 1, dt, f"missing {len(set(missing))} file(s)")
        return 1

    ok, note = True, "verify skipped"
    if not args.no_verify and targets:
        print("\nverifying against the committed results:")
        ok, lines = verify(targets)
        print("\n".join(lines))
        if ok:
            print("\nPASS: the analysis reproduces the committed results.")
        else:
            print("\nFAIL: the results DIFFER from the committed reference. "
                  "Nothing was modified - investigate before re-running.")
        note = "verify " + ("ok" if ok else "MISMATCH")
    elif not targets:
        print("\n(site.cases empty - nothing to verify against)")

    L.log_stage_run("step1_analysis", 0 if ok else 1, dt, note)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
