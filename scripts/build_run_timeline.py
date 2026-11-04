"""Build output/run_timeline.html: an interactive "post-game summary" of a run.

The timeline is the Dota-2 idea applied to the pipeline: the x axis is wall
time, each horizontal bar is one instrumented stage call (the "item"), and the
rows are the worker lanes (the "players").  Bars are coloured by phase, hover
gives the full span record, drag zooms, and clicking drills into the call tree.

Inputs (written by scripts/run_analysis.py, one pair per case):

    output/<label>/results/run_spans.csv    per-call timeline
    output/<label>/results/run_profile.csv  per-stage aggregates (inventory)

Output:

    output/run_timeline.html                self-contained, no network needed

Usage (from repo root, inside the venv):

    .venv\\Scripts\\python.exe scripts\\build_run_timeline.py
    .venv\\Scripts\\python.exe scripts\\build_run_timeline.py --check   # validate
    .venv\\Scripts\\python.exe scripts\\build_run_timeline.py --case az098_e903_t10
"""

import argparse
import datetime
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from solar_glare.profiling import PHASES, STAGE_SPECS  # noqa: E402
from solar_glare import viz  # noqa: E402  (for the shared phase palette)

OUT = os.path.join(ROOT, "output")
TEMPLATE = os.path.join(ROOT, "scripts", "run_timeline_template.html")
HTML = os.path.join(OUT, "run_timeline.html")
PAYLOAD_TOKEN = "__PAYLOAD__"


def case_labels(cfg_path):
    """Case labels in config order, so the picker matches site.cases."""
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return [c["label"] for c in (cfg.get("site", {}).get("cases") or [])
                if c.get("label")]
    except Exception:
        return []


def discover(labels):
    """Case label -> (spans path, profile path) for the cases that have both."""
    found = []
    for lbl in labels:
        sp = os.path.join(OUT, lbl, "results", "run_spans.csv")
        pf = os.path.join(OUT, lbl, "results", "run_profile.csv")
        if os.path.exists(sp) and os.path.exists(pf):
            found.append((lbl, sp, pf))
    return found


def build_case(label, spans_path, prof_path, t_off=0.0):
    df = pd.read_csv(spans_path)
    prof = pd.read_csv(prof_path)

    # run_spans.csv is appended when a stage CLOSES, so its row order is exit
    # order and `span_id` is NOT the row index.  Re-order chronologically and
    # renumber, so span_id == row index: the template can then index parents
    # directly (a span_id used as a row index would otherwise walk a cycle),
    # and t0 comes out sorted for the painter order and hit testing.
    df = df.sort_values(["t0", "t1", "span_id"], kind="stable").reset_index(drop=True)
    remap = {int(sid): new for new, sid in enumerate(df.span_id)}
    pid = [-1 if p < 0 else remap[int(p)] for p in df.parent_id]
    # a root has parent_id -1, so anything below -1 is a broken reference
    if any(p < -1 or p >= len(df) for p in pid):
        raise SystemExit("%s: parent_id refers to a span outside the file" % label)

    # -- lane index per row -------------------------------------------------
    # spans are appended when a stage CLOSES, so "first appearance" order is
    # exit order, not W1..W16; sort the worker labels numerically instead.
    lane_names = []
    for name in dict.fromkeys(df.thread_label):
        if name not in lane_names:
            lane_names.append(name)

    def lane_key(name):
        if name == "main":
            return (0, 0)
        return (1, int(name[1:])) if name[1:].isdigit() else (2, name)

    lane_names.sort(key=lane_key)
    lane_index_map = {n: i for i, n in enumerate(lane_names)}

    # -- shared index tables ------------------------------------------------
    stage_names = list(dict.fromkeys(df.stage))
    stage_ix = {s: i for i, s in enumerate(stage_names)}
    phase_names = list(PHASES)
    phase_ix = {p: i for i, p in enumerate(phase_names)}

    n = len(df)
    dur = (df.t1 - df.t0)
    # -- one continuous clock for the whole sweep ----------------------------
    # run_analysis.py runs the cases back to back, one at a time, and each case
    # restarts its own timer at 0.  Laying the cases out on the sweep's real
    # elapsed time (case k starts where case k-1 ended) is what the reader
    # actually wants from a timeline; giving every case its own 0 would make
    # six unrelated runs look like one six-lane parallel run.  Only the OFFSET
    # moves: dur, self_s and every parent/child relationship are untouched, so
    # the profiler's own time accounting still closes.
    t0n = df.t0 - df.t0.min()                  # each case normalised to its own 0
    t1n = (df.t1 - df.t0.min()) + t_off        # ...then moved onto the sweep clock
    t0n = t0n + t_off
    total_ms = float(t1n.max()) * 1000.0

    # inventory comes from the aggregate profile (the authoritative view)
    inv = []
    for r in prof.itertuples():
        inv.append({
            "stage": r.stage,
            "phase": r.phase,
            "calls": int(r.calls),
            "wall_ms": float(r.wall_s) * 1000.0,
            "self_ms": float(r.self_s) * 1000.0,
            "pct": 100.0 * float(r.wall_s) / float(prof.wall_s.sum()),
        })
    inv.sort(key=lambda r: -r["wall_ms"])
    inv_stage = {r["stage"]: r for r in inv}

    # per-phase span counts, for the legend
    phase_spans = {}
    for p in df.phase:
        phase_spans[p] = phase_spans.get(p, 0) + 1

    # key outputs that are NOT already stored elsewhere (same suppression rule
    # as the console table), so the tooltip can show them
    keys = {}
    for r in prof.itertuples():
        kt = getattr(r, "key_outputs", None)
        if isinstance(kt, str) and kt.strip():
            keys[r.stage] = kt.strip()

    # busiest single span, for the header
    top = int(dur.values.argmax())
    busiest = {"name": str(df.stage.iloc[top]), "ms": float(dur.iloc[top]) * 1000.0}

    par = None
    for r in prof.itertuples():
        if r.stage == "analyze_pair":
            par = "peak %d" % int(r.peak_concurrency) if pd.notna(r.peak_concurrency) else None
            break

    return {
        "label": label,
        "n": n,
        # where this case sits on the sweep clock, and how long it ran
        "t_off_ms": round(t_off * 1000.0, 3),
        "dur_ms": round(float((df.t1 - df.t0.min()).max()
                              - (df.t0 - df.t0.min()).min()) * 1000.0, 3),
        "total_ms": round(total_ms, 3),
        "n_threads": int(df.thread.nunique()),
        "n_lanes": len(lane_names),
        "max_depth": int(df.depth.max()),
        "nm": stage_names,
        "phn": phase_names,
        "id": list(range(len(df))),
        "pid": pid,
        "st": [stage_ix[s] for s in df.stage],
        "ph": [phase_ix[p] for p in df.phase],
        "th": [lane_index_map[t] for t in df.thread_label],
        "lane_name": lane_names,
        "dp": df.depth.astype(int).tolist(),
        "t0": [round(v * 1000.0, 3) for v in t0n],
        "t1": [round(v * 1000.0, 3) for v in t1n],
        "dur": [round(v * 1000.0, 3) for v in dur],
        "self": [round(v, 6) for v in df.self_s],
        "inv": inv,
        "phase_spans": phase_spans,
        "keys": keys,
        "busiest": busiest,
        "par": par,
    }


def validate(case):
    """Structural invariants of one case payload."""
    problems = []
    n = case["n"]
    nrows = len(case["lane_name"])
    by_id = {sid: i for i, sid in enumerate(case["id"])}
    if len(by_id) != n:
        problems.append("duplicate span_id (%d unique of %d)" % (len(by_id), n))
    # ids must be exactly 0..n-1, because the template indexes parents with them
    if case["id"] != list(range(n)):
        problems.append("span ids are not 0..n-1 (template parent walk would break)")
    if any(p < -1 or p >= n for p in case["pid"]):
        problems.append("parent_id outside [-1, n-1]")
    if any(case["t0"][i] > case["t0"][i + 1] for i in range(n - 1)):
        problems.append("t0 not sorted ascending")
    # walking parents must reach a root; a cycle would hang the browser
    for i in range(n):
        if case["dp"][i] == 0:
            continue
        hops, p = 0, i
        while case["pid"][p] != -1 and hops <= n:
            p = case["pid"][p]
            hops += 1
        if case["pid"][p] != -1:
            problems.append("span %d: parent chain does not terminate" % i)
            break
    # run_spans.csv is written with float_format="%.6f", i.e. 1 microsecond.
    # `dur` is stored as (t1 - t0) and `self_s` independently, so for a leaf
    # stage - where self_s IS the duration - the two can disagree by up to
    # 1 us after rounding.  Anything larger is a real inconsistency.
    round_tol = 2e-6
    for i in range(n):
        t0, t1 = case["t0"][i], case["t1"][i]
        if not (t1 >= t0):
            problems.append("span %d: t1 < t0" % case["id"][i])
        if case["self"][i] < -round_tol:
            problems.append("span %d: negative self_s" % case["id"][i])
        if case["self"][i] > (t1 - t0) / 1000.0 + round_tol:
            problems.append("span %d: self_s %.6f > duration %.6f"
                            % (case["id"][i], case["self"][i], (t1 - t0) / 1000.0))
        if not (0 <= case["th"][i] < nrows):
            problems.append("span %d: lane row out of range" % case["id"][i])
        p = case["pid"][i]
        if case["dp"][i] == 0:
            if p != -1:
                problems.append("span %d: root with parent %d" % (case["id"][i], p))
        else:
            if p not in by_id:
                problems.append("span %d: missing parent %d" % (case["id"][i], p))
            else:
                j = by_id[p]
                if case["dp"][j] != case["dp"][i] - 1:
                    problems.append("span %d: depth %d under parent depth %d"
                                    % (case["id"][i], case["dp"][i], case["dp"][j]))
                if case["th"][j] != case["th"][i]:
                    problems.append("span %d: parent on another lane" % case["id"][i])
                if case["t0"][j] > t0:
                    problems.append("span %d: parent starts later" % case["id"][i])
    # self_s + direct-children wall == wall.  This is the profiler's own time
    # accounting: a parent's self time is its wall time minus its children.
    # If it does not close, the spans cannot be trusted for the timeline.
    # run_spans.csv stores t0/t1/self_s rounded to 1 us, and the identity mixes
    # three independently rounded quantities: the parent's dur (t1 - t0, so up
    # to 1 us), the parent's own self_s (up to 0.5 us) and each child's dur
    # (again t1 - t0, up to 1 us).  The residual is therefore bounded by
    # 1.5 us + 1 us per child - not by a constant, and not by 0.5 us per child:
    # a child's dur is a DIFFERENCE of two rounded values, so it carries twice
    # what a stored value does.  (Observed 2026-09-28: a 139 us span with one
    # child closed to within 2.0 us, which the old 1.0+0.5/child bound rejected
    # although it sits inside the true 2.5 us bound.)
    kids, nkids_of = {}, {}
    for i in range(n):
        p = case["pid"][i]
        if p >= 0:
            kids[p] = kids.get(p, 0.0) + case["dur"][i]
            nkids_of[p] = nkids_of.get(p, 0) + 1
    for i in range(n):
        nkids = nkids_of.get(i, 0)
        tol_ms = (1.5 + nkids) * 1e-3
        want = case["dur"][i]
        got = kids.get(i, 0.0) + case["self"][i] * 1000.0
        if abs(got - want) > tol_ms:
            problems.append("span %d: self+children %.6f ms != wall %.6f ms "
                            "(tol %.4f ms over %d children)"
                            % (case["id"][i], got, want, tol_ms, nkids))
            break

    if case["max_depth"] < 0:
        problems.append("negative max_depth")
    return problems


def validate_sweep(cases):
    """The merged axis is ONE continuous serial clock.

    run_analysis.py runs the cases back to back, so case k must begin exactly
    where case k-1 ended: no overlap (they never ran at the same time) and no
    invented gap (the only real gaps are the sub-second of uninstrumented work
    between cases, which the spans do not record).  This is the invariant the
    whole merged layout rests on, so --check proves it rather than trusting it.
    """
    problems = []
    off = 0.0
    for c in cases:
        got = c["t_off_ms"] / 1000.0
        if abs(got - off) > 1e-6:
            problems.append("%s: starts at %.3f s, expected %.3f s (cases must "
                            "be contiguous on the sweep clock)"
                            % (c["label"], got, off))
        # every span must sit inside this case's own slice of the clock
        lo, hi = c["t_off_ms"], c["t_off_ms"] + c["dur_ms"]
        if min(c["t0"]) < lo - 1e-3:
            problems.append("%s: a span starts at %.3f s, before the case's own "
                            "%.3f s" % (c["label"], min(c["t0"]) / 1000.0, lo / 1000.0))
        if max(c["t1"]) > hi + 1e-3:
            problems.append("%s: a span ends at %.3f s, after the case's own "
                            "%.3f s" % (c["label"], max(c["t1"]) / 1000.0, hi / 1000.0))
        off += c["dur_ms"] / 1000.0
    if cases and abs(cases[-1]["total_ms"] - off * 1000.0) > 1e-3:
        problems.append("the last case ends at %.3f s but the sweep is %.3f s"
                        % (cases[-1]["total_ms"] / 1000.0, off))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", metavar="LABEL", default=None,
                    help="only this case (default: every case with span data)")
    ap.add_argument("--check", action="store_true",
                    help="validate the payload structure and invariants, then stop")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "site.yaml"))
    args = ap.parse_args()

    labels = case_labels(args.config)
    found = discover(labels)
    if args.case:
        found = [f for f in found if f[0] == args.case]
        if not found:
            raise SystemExit("no run_spans.csv + run_profile.csv for case %r "
                             "(run scripts/run_analysis.py first)" % args.case)
    if not found:
        raise SystemExit("no cases with run_spans.csv under output/ "
                         "(run scripts/run_analysis.py first)")

    # chain the cases onto one continuous clock: case k starts where k-1 ended
    cases, off = [], 0.0
    for lbl, sp, pf in found:
        c = build_case(lbl, sp, pf, t_off=off)
        cases.append(c)
        off += c["dur_ms"] / 1000.0

    bad = 0
    for c in cases:
        probs = validate(c)
        if probs:
            bad += 1
            print("[FAIL] %s: %d problem(s)" % (c["label"], len(probs)))
            for p in probs[:10]:
                print("       -", p)
        else:
            print("[ok]   %-16s %6d spans, %2d lanes, depth<=%d, %7.1f s"
                  "  (sweep %7.1f -> %7.1f s)"
                  % (c["label"], c["n"], c["n_lanes"], c["max_depth"],
                     c["dur_ms"] / 1000.0, c["t_off_ms"] / 1000.0,
                     c["total_ms"] / 1000.0))
    probs = validate_sweep(cases)
    if probs:
        bad += 1
        print("[FAIL] sweep clock: %d problem(s)" % len(probs))
        for p in probs[:10]:
            print("       -", p)
    else:
        print("[ok]   sweep clock: %d cases contiguous, %s s total"
              % (len(cases), ("%.1f" % (off)) if cases else "0"))
    if bad:
        raise SystemExit("%d problem(s) failed validation" % bad)

    payload = {
        "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "phases": list(PHASES),
        "phase_colors": viz._PROFILE_PHASE_COLORS,
        "cases": cases,
    }

    if args.check:
        print("payload OK (%d case(s)); not writing (--check)" % len(cases))
        return

    with open(TEMPLATE, encoding="utf-8") as fh:
        html = fh.read()
    if PAYLOAD_TOKEN not in html:
        raise SystemExit("template lost its %s token" % PAYLOAD_TOKEN)
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    html = html.replace(PAYLOAD_TOKEN, blob)
    with open(HTML, "w", encoding="utf-8") as fh:
        fh.write(html)

    total = sum(c["n"] for c in cases)
    print("wrote %s (%.1f MB, %d cases, %s spans, %s of payload)"
          % (os.path.relpath(HTML, ROOT), len(html) / 1e6, len(cases),
             f"{total:,}", f"{len(blob)/1e6:.1f} MB"))


if __name__ == "__main__":
    main()
