"""Shared plumbing for the three workflow stages (scripts/stages/step1..3, run_all).

The stages are THIN DRIVERS.  None of them contains any physics: they launch the
existing scripts as subprocesses and read back what those scripts wrote.  That is
deliberate, and it is the reason a stage run cannot change a result:

  * every stage calls scripts/<something>.py with sys.executable, i.e. the same
    interpreter (.venv) that the user launched, so the dependency set is the one
    in requirements.txt and nothing else;
  * every stage runs with cwd=repo root, which is where all those scripts assume
    they are (they resolve their own paths from __file__, but the output they
    read and write is relative to the root);
  * each script is a SEPARATE PROCESS, so the module-level state of one
    validator can never leak into the next one.  This matters here: several
    validators (validate_charts, validate_kuantan) parse
    sys.argv at IMPORT time, so importing them in-process would silently apply
    this run's flags to them.

This module therefore depends on the standard library only (csv, os, subprocess,
time, yaml).  It deliberately does not import pandas or solar_glare: reading a
summary.csv with csv.DictReader is enough, and a stage driver that imports the
heavy pipeline would slow down every invocation for nothing.

One audit trail is written: output/validation/stage_runs.txt, one line per stage
run (timestamp, stage, exit code, wall time, free-text note).  It is separate
from the per-validator logs because those are appended to by the validators
themselves and this file answers a different question: what was run, when, and
did it pass.
"""

import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

# scripts/stages/<file>.py -> scripts/stages -> scripts -> repo root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "output")
VALIDATION = os.path.join(OUT, "validation")
STAGE_LOG = os.path.join(VALIDATION, "stage_runs.txt")
CONFIG = os.path.join(ROOT, "config", "site.yaml")

PY = sys.executable

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


# --------------------------------------------------------------------------
# committed results: the reference this project's numbers are checked against
# --------------------------------------------------------------------------
# Source of truth: AGENTS.md ("Headline results") and the committed
# output/<label>/results/summary.csv of the six-case sweep.  A stage run that
# reproduces these reproduces the published analysis.
#
# The five integer columns (G/Y/R/minutes/IEC) are compared EXACTLY.  The peak
# is compared with a relative tolerance, because the number published in
# AGENTS.md carries only 3-4 significant digits while the CSV carries full
# double precision; 1e-3 is far tighter than that and far looser than the
# float noise the pipeline can produce.
#
# Re-based 2026-09-28 when `groups` went 18 -> 19 (config/site.yaml, the real
# module layout). The peaks are the ONLY values that move: the Beer-Lambert
# transmittance is evaluated at the centre of the sub-panel that wins each
# minute, so re-partitioning the 4,890 m row shifts the winner's centre and the
# annual peak E_r by <0.2% (largest: az278_e903_t10/west1_route_1, 0.846091 ->
# 0.847394 = +0.154%, which is what first tripped the 1e-3 tolerance). The five
# integer columns are unchanged - same hit minutes, same G/Y/R, same IEC group.
PEAK_RTOL = 1e-3

COMMITTED = [
    # case,              receptor,          G,   Y,   R, minutes, IEC,   peak W/cm2
    ("az098_e903_t10", "obs_ptt",          0, 489,  0, 489, "RG3", 1.1088408403919785),
    ("az098_e903_t10", "west1_route_11",   0, 351,  0, 351, "RG3", 0.6025518944705084),
    ("az098_e903_t15", "obs_ptt",          0, 466,  0, 466, "RG3", 0.8257200540235704),
    ("az098_e903_t15", "west1_route_11",   0, 336,  0, 336, "RG2", 0.4465154303971285),
    ("az098_e903_t20", "obs_ptt",          0, 470,  0, 470, "RG2", 0.5690899809018621),
    ("az098_e903_t20", "west1_route_11",   0, 327,  0, 327, "RG2", 0.301232732665228),
    ("az278_e903_t10", "west1_route_1",  152,   0,  0, 152, "RG2", 0.847394479845549),
    ("az278_e903_t10", "west1_route_2",    0, 104,  0, 104, "RG2", 0.9593948669311344),
    ("az278_e903_t15", "west1_route_1",   77,   0,  0,  77, "RG1", 0.4403909953896631),
]

# Per-case G/Y/R sums.  These close the loop on the nine rows above: if any
# OTHER receptor started to report glare, the per-case sums would move, so the
# check fails even though every committed row still matches.
EXPECTED_TOTALS = {
    "az098_e903_t10": (0, 840, 0),
    "az098_e903_t15": (0, 802, 0),
    "az098_e903_t20": (0, 797, 0),
    "az278_e903_t10": (152, 104, 0),
    "az278_e903_t15": (77, 0, 0),
    "az278_e903_t20": (0, 0, 0),
}
SWEEP_TOTALS = (229, 2543, 0)          # G 229 + Y 2543 + R 0 over the six cases
EXPECTED_CASES = list(EXPECTED_TOTALS)


# --------------------------------------------------------------------------
# process plumbing
# --------------------------------------------------------------------------
@dataclass
class StepResult:
    name: str
    cmd: list = field(default_factory=list)
    rc: int = 0
    seconds: float = 0.0
    status: str = PASS
    note: str = ""


def rel(path):
    """Repo-relative path, for messages that should not leak an absolute path."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def banner(number, title, subtitle=""):
    line = "=" * 78
    print()
    print(line)
    print(f"STEP {number}: {title}")
    if subtitle:
        print(subtitle)
    print(line)
    sys.stdout.flush()


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def script_path(script):
    """Resolve a script name to scripts/stages/ or scripts/.

    A stage script may drive another stage script (run_all.py -> step3...) as
    well as the project's own scripts, so both directories are searched, stages
    first.  Raises FileNotFoundError naming both places tried, because a wrong
    path here surfaces as Python's "can't open file" from the child, which does
    not say where it looked.
    """
    tried = []
    for base in (STAGE_DIR, SCRIPTS):
        p = os.path.join(base, script)
        tried.append(rel(p))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"{script} not found - tried {', '.join(tried)}")


def child_env():
    """Environment for a child script, with a UTF-8 stdout.

    This is not cosmetic.  The Windows console here defaults to codepage 874,
    and a validator that prints a Greek letter - validate_norm_vs_code.py
    checks that the text of BS EN 62471 really contains "alpha_max = 0.1 radian"
    and prints the needle verbatim - dies with UnicodeEncodeError the moment it
    writes that line.  The crash is in the console, not in the check: the same
    script reports 100 passed / 0 failed once stdout can encode the character.
    A stage runner that reports rc=1 there would be reporting a Windows
    codepage as a validation failure, so the encoding is pinned here, in the
    layer that decides how children are launched, and the validators themselves
    stay untouched.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_script(script, args=(), log=None):
    """Run scripts/<script> as a child process, streaming its output live.

    Returns (rc, seconds).  When `log` is given and exists, the byte size
    before/after is measured too, because most validators APPEND to their log
    instead of rewriting it - a run therefore grows the file, and a stage report
    that stays silent about that would read as "the log is current" when it is
    actually a stack of runs.
    """
    cmd = [PY, script_path(script)] + list(args)
    size0 = os.path.getsize(log) if log and os.path.exists(log) else None
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env())
    proc.wait()
    dt = time.time() - t0
    print(f"--- {script} {' '.join(args)} -> rc={proc.returncode} in {dt:.1f}s")
    if log and size0 is not None:
        size1 = os.path.getsize(log)
        print(f"    log {rel(log)} now {size1:,} bytes "
              f"(was {size0:,}; most validators append rather than rewrite)")
    elif log:
        print(f"    log {rel(log)} written")
    sys.stdout.flush()
    return proc.returncode, dt


def run_step(name, script, args=(), warn_only=False, log=None, precheck=None):
    """Run one child script and classify the outcome as a StepResult."""
    if precheck is not None:
        blocked = precheck()
        if blocked:
            print(f"--- {name}: SKIP ({blocked})")
            return StepResult(name, [script, *args], 0, 0.0, SKIP, str(blocked))
    rc, dt = run_script(script, args, log=log)
    if rc == 0:
        status = PASS
    else:
        status = WARN if warn_only else FAIL
    return StepResult(name, [script, *args], rc, dt, status)


def table(results, width=26):
    """Fixed-width PASS/FAIL table for a list of StepResult."""
    if not results:
        return "(nothing ran)"
    name_w = max(width, max(len(r.name) for r in results) + 2)
    out = [f"{'stage':<{name_w}}{'status':<8}{'rc':>4}  {'time':>8}  note",
           "-" * (name_w + 34)]
    for r in results:
        note = r.note if len(r.note) <= 46 else r.note[:43] + "..."
        out.append(f"{r.name:<{name_w}}{r.status:<8}{r.rc:>4}  "
                   f"{r.seconds:>7.1f}s  {note}")
    return "\n".join(out)


def tally(results):
    counts = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


def exit_code(results):
    """0 unless a stage actually FAILED.  WARN and SKIP never fail a stage."""
    return 1 if any(r.status == FAIL for r in results) else 0


# --------------------------------------------------------------------------
# audit trail
# --------------------------------------------------------------------------
def log_stage_run(stage, rc, seconds, note=""):
    os.makedirs(VALIDATION, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(STAGE_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp} | {stage:<16} | rc={rc} | {seconds:7.1f}s | {note}\n")


# --------------------------------------------------------------------------
# config + result readers
# --------------------------------------------------------------------------
def case_labels(config=CONFIG):
    """Case labels in config order, read straight from site.yaml."""
    import yaml
    with open(config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return [c["label"] for c in (cfg.get("site", {}).get("cases") or [])
            if c.get("label")]


def summary_path(case):
    return os.path.join(OUT, case, "results", "summary.csv")


def read_summary(case):
    """summary.csv as a list of dicts, or (None, path) if it is not there."""
    path = summary_path(case)
    if not os.path.exists(path):
        return None, path
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh)), path


def case_totals(rows):
    return (sum(int(r["G"]) for r in rows),
            sum(int(r["Y"]) for r in rows),
            sum(int(r["R"]) for r in rows))


def glare_rows(rows):
    return [r for r in rows if int(r["minutes_with_glare"]) > 0]
