"""STEP 4 of 4 - build the HTML report.

    .venv\\Scripts\\python.exe scripts\\stages\\step4_report.py

Builds output/report.html from output/report.md, embedding every figure as a
base64 data URI, so the report is one self-contained file that opens offline and
can be mailed as-is.

output/report.md is the hand-maintained Thai report: its prose, its tables and
its 96 figure references are written by a person, not generated.  That is why
this stage BUILDS but never EDITS it, and why it emits warnings instead of
fixing things:

  * if report.md is older than the newest summary.csv, the numbers quoted in
    the prose may no longer match the analysis - the stage says so and leaves
    the text alone.  Refreshing those numbers is an editorial task, and
    AGENTS.md keeps that sequence explicit (run_analysis -> refresh report.md ->
    build_report_html.py);
  * every `![...](path)` reference in report.md is resolved against
    output/, and any that does not exist is listed.  build_report_html.py turns
    those into a visible "[missing ...]" marker, so they would otherwise be easy
    to miss; the stage counts them and fails the run if any are left.

After building, the generated page is re-read to confirm it carries the images
and no unresolved marker.

Runtime: ~5 s (the current page is ~14 MB, almost all of it base64).
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stage_lib as L  # noqa: E402

MD = os.path.join(L.OUT, "report.md")
HTML = os.path.join(L.OUT, "report.html")
MIN_HTML_BYTES = 1_000_000        # the current report embeds 96 PNGs (~14 MB)
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def main():
    L.banner(4, "REPORT",
             "Builds output/report.html (self-contained) from the hand-written "
             "output/report.md.")

    if not os.path.exists(MD):
        print(f"ERROR: {L.rel(MD)} not found - there is no report source to "
              f"build from.")
        L.log_stage_run("step4_report", 1, 0.0, "report.md missing")
        return 1

    md = open(MD, encoding="utf-8").read()
    refs = IMG_RE.findall(md)
    missing = [r for r in refs if not os.path.exists(os.path.join(L.OUT, r))]
    print(f"report.md : {L.rel(MD)} ({os.path.getsize(MD):,} bytes, "
          f"{len(refs)} figure references)")
    if missing:
        print(f"\nERROR: {len(missing)} figure reference(s) in report.md do not "
              f"exist under output/ - these would render as [missing ...]:")
        for r in missing[:20]:
            print(f"  - {r}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        L.log_stage_run("step4_report", 1, 0.0,
                        f"{len(missing)} missing figure ref(s)")
        return 1
    print("figures   : all present")

    # Staleness: the newest per-case result is the floor the prose should meet.
    cases = L.case_labels()
    newest, newest_case = 0.0, ""
    for case in cases:
        t = L.mtime(L.summary_path(case))
        if t and t > newest:
            newest, newest_case = t, case
    md_t = L.mtime(MD)
    if newest and md_t and md_t < newest:
        print(f"\nWARNING: report.md is older than the newest results "
              f"(output/{newest_case}/results/summary.csv). The numbers in the "
              f"prose may not match the latest run.")
        print("         Refresh the numbers in report.md, then re-run this "
              "step. Nothing is edited automatically.")
    elif newest and md_t:
        print(f"freshness : report.md is newer than every summary.csv "
              f"(newest result = output/{newest_case})")

    results = [L.run_step("build_report", "build_report_html.py", [])]
    if results[0].status == L.FAIL:
        L.log_stage_run("step4_report", 1, results[0].seconds, "build failed")
        return 1

    if not os.path.exists(HTML):
        print(f"\nERROR: {L.rel(HTML)} was not written.")
        L.log_stage_run("step4_report", 1, results[0].seconds, "no html written")
        return 1

    page = open(HTML, encoding="utf-8").read()
    n_img = page.count('<img src="data:image')
    n_missing = page.count("[missing")
    size = len(page.encode("utf-8"))
    print(f"    {L.rel(HTML)} = {size:,} bytes, {n_img} embedded image(s)")
    if n_missing:
        print(f"\nERROR: the built page still contains {n_missing} "
              f"[missing ...] marker(s).")
        L.log_stage_run("step4_report", 1, results[0].seconds,
                        f"{n_missing} missing marker(s) in html")
        return 1
    if n_img != len(refs):
        print(f"\nWARNING: report.md has {len(refs)} figure references but the "
              f"page embeds {n_img}.")
    if size < MIN_HTML_BYTES:
        print(f"\nWARNING: {size:,} bytes is below the {MIN_HTML_BYTES:,} "
              f"expected for a fully embedded report.")

    results[0].note = f"{size:,} B, {n_img} images"
    print("\n" + L.table(results))
    print("open: " + L.rel(HTML))
    rc = L.exit_code(results)
    L.log_stage_run("step4_report", rc, results[0].seconds,
                    f"{size:,} B, {n_img} images")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
