"""Validate output/run_timeline.html -> output/validation/run_timeline.txt

Two layers:

  [A] Pure-Python checks on the file itself (no external tooling):
      the __PAYLOAD__ token was substituted, the payload is valid JSON, the
      cases are exactly the committed `site.cases` labels in config order, and
      the page carries no remote references (it must work offline).

  [B] scripts/validate_timeline_html.js, run under Node: extracts the page's
      own <script>, executes it against a stubbed DOM, drives every
      interaction, and re-checks the span data.  This is the only layer that
      can catch a JS regression -- e.g. ancestry() walking a parent cycle and
      hanging the browser.  Node is NOT a project dependency: if it is missing
      the JS layer is reported as [SKIP], never as a pass.

Usage (from repo root, inside the venv):

    .venv\\Scripts\\python.exe scripts\\validate_timeline.py
    .venv\\Scripts\\python.exe scripts\\validate_timeline.py --html PATH
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
DEFAULT_HTML = os.path.join(OUT, "run_timeline.html")
JS = os.path.join(ROOT, "scripts", "validate_timeline_html.js")
REPORT = os.path.join(OUT, "validation", "run_timeline.txt")
CONFIG = os.path.join(ROOT, "config", "site.yaml")

PAYLOAD_RE = re.compile(
    r'<script id="payload" type="application/json">(.*?)</script>', re.S)

_lines = []


def out(line=""):
    print(line)
    _lines.append(line)


def case_labels():
    """Committed case labels, in config order (same order the builder uses)."""
    try:
        import yaml
        with open(CONFIG, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return [c["label"] for c in (cfg.get("site", {}).get("cases") or [])
                if c.get("label")]
    except Exception as exc:                       # pragma: no cover
        out("       (could not read config/site.yaml: %s)" % exc)
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", default=DEFAULT_HTML,
                    help="timeline to validate (default: output/run_timeline.html)")
    args = ap.parse_args()

    html_path = args.html
    if not os.path.exists(html_path):
        raise SystemExit("%s not found -- run scripts/run_analysis.py then "
                         "scripts/build_run_timeline.py first" % html_path)
    html = open(html_path, encoding="utf-8").read()

    out("=== run_timeline.html validation ===")
    out("file   : %s (%.2f MB)" % (os.path.relpath(html_path, ROOT), len(html) / 1e6))
    out("date   : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    out()

    passed = failed = skipped = 0

    def check(name, ok, detail=""):
        nonlocal passed, failed
        if ok:
            passed += 1
            out("  [OK ] %s%s" % (name, ("  " + detail) if detail else ""))
        else:
            failed += 1
            out("  [FAIL] %s%s" % (name, ("  " + detail) if detail else ""))

    # ---------------------------------------------------------------- [A]
    out("[A] file structure")
    check("A1 no unsubstituted __PAYLOAD__ token", "__PAYLOAD__" not in html)
    m = PAYLOAD_RE.search(html)
    check("A2 payload script block present", bool(m))
    if not m:
        out("\nRESULT: %d passed, %d failed" % (passed, failed))
        _write()
        return 1

    try:
        data = json.loads(m.group(1))
        parsed = True
    except Exception as exc:
        data, parsed = None, False
        out("       (json error: %s)" % exc)
    check("A3 payload is valid JSON", parsed)
    if not parsed:
        out("\nRESULT: %d passed, %d failed" % (passed, failed))
        _write()
        return 1

    payload_cases = [c.get("label") for c in data.get("cases", [])]
    check("A4 payload has at least one case", len(payload_cases) >= 1,
          "%d case(s)" % len(payload_cases))

    labels = case_labels()
    if labels:
        check("A5 payload cases == config site.cases, in order",
              payload_cases == labels,
              "" if payload_cases == labels else
              "payload %s vs config %s" % (payload_cases, labels))
    else:
        skipped += 1
        out("  [SKIP] A5 config cross-check (site.cases unreadable)")

    n_spans = sum(int(c.get("n", 0)) for c in data.get("cases", []))
    out("       %d spans across %d case(s)" % (n_spans, len(payload_cases)))

    remote = re.findall(r"https?://", html)
    check("A6 self-contained (no remote references)", not remote,
          "" if not remote else "%d http(s) reference(s)" % len(remote))
    check("A7 no external <script src=>", not re.findall(r"<script[^>]*\ssrc=", html))

    # ---------------------------------------------------------------- [B]
    out()
    out("[B] page behaviour (Node vm + stubbed DOM)")
    node = shutil.which("node") or shutil.which("nodejs")
    if not node:
        # deliberately NOT a pass: the JS layer simply did not run
        skipped += 1
        out("  [SKIP] B* Node not found on PATH -- install Node.js to run the "
            "JS layer (scripts/validate_timeline_html.js)")
        out("         The [A] checks above still cover the payload; the JS "
            "layer is what catches e.g. a parent-walk cycle that hangs the page.")
    else:
        proc = subprocess.run([node, JS, html_path], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        for line in (proc.stdout or "").rstrip().splitlines():
            out("  " + line if not line.startswith("  ") else line)
        if proc.returncode == 0:
            passed += 1
            out("  [OK ] B* all page-behaviour checks passed")
        else:
            failed += 1
            out("  [FAIL] B* page-behaviour checks failed (node exit %d)"
                % proc.returncode)
        if proc.stderr and proc.stderr.strip():
            out("  node stderr: " + proc.stderr.strip().replace("\n", "\n  "))

    out()
    verdict = ("RESULT: %d passed, %d failed" % (passed, failed)
               + (", %d skipped" % skipped if skipped else ""))
    out("=== " + verdict + " ===")
    _write(verdict)
    return 1 if failed else 0


def _write(verdict=None):
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("\nwrote %s" % os.path.relpath(REPORT, ROOT))


if __name__ == "__main__":
    sys.exit(main())
