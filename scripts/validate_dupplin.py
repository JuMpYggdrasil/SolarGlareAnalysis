"""External benchmark: Dupplin Solar (Scotland) ForgeSolar/SGHAT report.

Source document: "Glare Analysis - Dupplin Solar v1", BLC Energy, created
08 Jan 2026 / updated 16 Jan 2026 (references/Dupplin_Solar_ForgeSolar_Glare_2026.pdf).
Geometry, settings and results are published IN FULL in that report, so unlike
the Kuantan benchmark (where the paper withholds the ForgeSolar geometry) this
case is a 1:1 reconstruction from config/duplin_scotland.yaml.

Published annual results (glare minutes per receptor):
  A9       : Green 7     + Yellow 1,051
  Roman Rd : Green 267   + Yellow 2,892
  Rural Rd : Green 514   + Yellow 2,226
  OP 13    : Green 123   + Yellow 0
  All other OP 1-18 (except 13): no glare
  Array total: Green 911 + Yellow 6,169 = 7,080 min/yr

Reconstruction settings (config/duplin_scotland.yaml):
  site 56.376594 N, -3.544063 E (polygon centroid), alt 120 m, UTC, year 2026
  PV fixed tilt 20 deg, azimuth 180 deg (true south), height 2.67 m AGL,
  909 x 1333 m rectangle (area ~1.21 km2, 16 groups along E-W long axis),
  smooth glass w/ ARC, reflectivity 0.08, slope error 8.43 mrad (SGHAT v6
  Table 2 ARC correlate), variable DNI (Ineichen clear-sky; the report's
  "DNI peaks at 1000 W/m2" is a VARIABLE profile, not constant 1000), min_dni=0.
  Receptors: OP 1-18 (1.5-4 m AGL) + every vertex of routes A9 (22), Roman Rd
  (18), Rural Rd (16) at 1.5 m AGL. Route totals are the per-minute UNION over
  the route's vertices (worst hazard wins per minute), matching how SGHAT
  aggregates a path receptor.

KNOWN MODEL LIMITATIONS (written to the output file):
  - The report models the array as a 37-vertex polygon of module rows; we use
    one flat SOLID rectangle of equal area aligned to the E-W long axis. A
    solid sheet over-states the reflective area seen by receptors sitting on
    or near the array boundary, which is the dominant cause of the systematic
    ~4-6x over-prediction in glare minutes.
  - The report includes 17 obstruction/hedgerow components that block glare;
    this pipeline has no obstruction model, so computed minutes can exceed
    the report's numbers (especially at OPs the report reports as glare-free).
  - Route receptors are two-way paths with a 50 deg azimuthal view angle in
    the report; we sample each vertex as a discrete point (no view-angle
    gate), so route totals are approximated by the per-vertex union.
  - The report uses ForgeSolar's glare-spot (V2) method over the polygon; we
    use the SGHAT v6 angular approximation (V1-style) over one equivalent
    panel, which inflates near-field hits.
  - Comparison is therefore ORDER-OF-MAGNITUDE + BEHAVIOURAL per receptor,
    not an exact-number match. Behavioural agreement (no Red, Yellow >> Green,
    glare only on the same receptors/routes, OP13 hit, morning/afternoon
    timing) is the meaningful validation result here.

Usage:
  .venv\\Scripts\\python.exe scripts\\validate_dupplin.py
Writes output/validation/dupplin.txt.
"""

import os
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from solar_glare.analysis import load_config, run_analysis

OUT_PATH = os.path.join(REPO, "output", "validation", "dupplin.txt")
CFG_PATH = os.path.join(REPO, "config", "duplin_scotland.yaml")

HITMODE = "circular"
for i, a in enumerate(sys.argv):
    if a == "--hit-test" and i + 1 < len(sys.argv):
        HITMODE = sys.argv[i + 1]
        break

# Report reference results (glare minutes).
PAPER = {
    "A9": {"G": 7, "Y": 1051, "total": 1058},
    "RomanRd": {"G": 267, "Y": 2892, "total": 3159},
    "RuralRd": {"G": 514, "Y": 2226, "total": 2740},
    "OP13": {"G": 123, "Y": 0, "total": 123},
    "total": {"G": 911, "Y": 6169, "total": 7080},
}

ROUTE_PREFIXES = {"A9": "A9_", "RomanRd": "RomanRd_", "RuralRd": "RuralRd_"}

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        tag = "OK "
    else:
        FAIL += 1
        tag = "FAIL"
    line = f"  [{tag}] {name}" + (f"  -- {detail}" if detail else "")
    print(line)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _count(detail):
    """G/Y/R minute counts from a glare-only detail frame (one row per minute)."""
    if detail.empty:
        return {"G": 0, "Y": 0, "R": 0, "total": 0}
    return {
        "G": int((detail["hazard"] == "G").sum()),
        "Y": int((detail["hazard"] == "Y").sum()),
        "R": int((detail["hazard"] == "R").sum()),
        "total": int(len(detail)),
    }


def route_union(details, prefix):
    """Per-minute UNION over all route vertices (worst hazard wins per minute)."""
    frames = [d for (a, r), d in details.items()
              if r.startswith(prefix) and not d.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    rank = df["hazard"].map({"N": 0, "G": 1, "Y": 2, "R": 3})
    df = df.assign(_rank=rank)
    df = (df.sort_values(["_rank", "retinal_wcm2"], ascending=[False, False])
          .drop_duplicates(subset=["time"], keep="first")
          .drop(columns="_rank"))
    return df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    t_start = time.time()
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n# ---- validate_dupplin.py run {pd.Timestamp.now()} ----\n")
        f.write("\nPaper reference (BLC Energy ForgeSolar report, Dupplin Solar,\n"
                "created 08/16 Jan 2026; references/Dupplin_Solar_ForgeSolar_Glare_2026.pdf):\n"
                "  A9: G=7 Y=1051 | Roman Rd: G=267 Y=2892 | Rural Rd: G=514 Y=2226\n"
                "  OP13: G=123 Y=0 | all other OPs: 0\n"
                "  Array total: G=911 Y=6169 = 7080 min/yr\n"
                "Reconstruction (config/duplin_scotland.yaml):\n"
                "  site 56.376594N 3.544063W alt 120 m (UTC, 2026, 1-min)\n"
                "  PV fixed tilt 20 deg, az 180 deg, 909x1333 m rectangle ~1.21 km2,\n"
                "  groups=16, smooth glass w/ ARC, reflectivity 0.08,\n"
                "  slope error 8.43 mrad (SGHAT v6 Table 2 ARC correlate),\n"
                "  DNI variable (Ineichen clear-sky; report 'DNI peaks at 1000'), min_dni=0\n"
                "  OPs 1-18 + all route vertices (A9/Roman Rd/Rural Rd) at 1.5 m AGL;\n"
                "  route totals = per-minute union over the route's vertices\n"
                "Limitations: polygon->solid rectangle (rows+gaps not modelled), so\n"
                "  boundary receptors see an over-large reflective sheet; no obstruction\n"
                "  model (17 report components ignored); no 50-deg route view angle;\n"
                "  V1-style angular method vs report's glare-spot V2. These inflate our\n"
                "  minutes ~4-6x; interpret behaviour + order-of-magnitude only.\n")

    cfg = load_config(CFG_PATH)
    cfg["environment"]["hit_test_mode"] = HITMODE
    t0 = time.time()
    _, _, _, long_df, summary, details, site, arrays, receptors = run_analysis(cfg)
    run_s = time.time() - t0
    print(f"full-year per-minute run in {run_s:.0f}s (hit-test={HITMODE})")
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n== [1] Full-year per-minute run in {run_s:.0f}s "
                f"({len(arrays)} arrays x {len(receptors)} receptors, "
                f"hit-test={HITMODE}) ==\n")

    # ---- per-receptor computed totals ----
    computed = {}
    for (a, r), d in details.items():
        computed[r] = _count(d)
    for prefix in ROUTE_PREFIXES.values():
        pass  # routes aggregated separately below

    # summary index: (receptor, array); grab OP totals from summary
    print("\n== [2] Computed vs report (minutes/yr) ==")
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write("\n== [2] Computed vs report (minutes/yr) ==\n")

    rows = []
    op_names = [r.name for r in receptors if not any(
        r.name.startswith(p) for p in ROUTE_PREFIXES.values())]
    for rname in op_names:
        sub = summary.xs(rname, level="receptor")
        rows.append((rname, {
            "G": int(sub["G"].sum()), "Y": int(sub["Y"].sum()),
            "R": int(sub["R"].sum()),
            "total": int(sub[["G", "Y", "R"]].sum().sum()),
        }))

    total = {"G": 0, "Y": 0, "R": 0, "total": 0}
    for rname, c in rows:
        total["G"] += c["G"]
        total["Y"] += c["Y"]
        total["R"] += c["R"]
        total["total"] += c["total"]

    for prefix, label in [("A9_", "A9"), ("RomanRd_", "Roman Rd"), ("RuralRd_", "Rural Road")]:
        c = _count(route_union(details, prefix))
        rows.append((label, c))
        total["G"] += c["G"]
        total["Y"] += c["Y"]
        total["R"] += c["R"]
        total["total"] += c["total"]

    for rname, c in rows:
        line = f"  {rname:<16} G={c['G']:6d} Y={c['Y']:6d} R={c['R']:6d} tot={c['total']:7d}"
        print(line)
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    line = (f"  {'ARRAY TOTAL':<16} G={total['G']:6d} Y={total['Y']:6d} "
            f"R={total['R']:6d} tot={total['total']:7d}")
    print(line)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    # ---- behavioural / order-of-magnitude checks ----
    print("\n== [3] Behavioural + order-of-magnitude checks vs report ==")
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write("\n== [3] Behavioural + order-of-magnitude checks vs report ==\n")

    computed_by_name = dict(rows)

    ref_total = PAPER["total"]["total"]
    ratio = total["total"] / ref_total
    check("array total within order of magnitude (0.3x..3x of 7,080 min)",
          0.3 <= ratio <= 3.0,
          f"ours={total['total']} min, paper={ref_total} min (ratio={ratio:.2f})")

    check("no Red glare (report has G/Y only)",
          total["R"] == 0, f"ours R={total['R']} (paper R=0)")

    check("Yellow dominates Green like the report",
          total["Y"] >= total["G"],
          f"ours G={total['G']} Y={total['Y']}; paper G=911 Y=6169")

    for key, label in [("A9", "A9"), ("RomanRd", "Roman Rd"), ("RuralRd", "Rural Road")]:
        ours = computed_by_name[label]
        ref = PAPER[key]["total"]
        r = ours["total"] / ref if ref else 0.0
        check(f"route {label} glare within order of magnitude",
              0.3 <= r <= 3.0, f"ours={ours['total']} min, paper={ref} min (ratio={r:.2f})")

    op13 = computed_by_name["OP13"]
    check("OP13 shows glare like the report (report: G123/Y0)",
          op13["total"] > 0 and op13["R"] == 0,
          f"ours G={op13['G']} Y={op13['Y']} R={op13['R']}; paper G=123 Y=0")

    # OPs the report says are glare-free: our no-obstruction model may see glare,
    # so report as informational only, not a pass/fail check.
    op13_others = [c for n, c in rows if n.startswith("OP") and n != "OP13"]
    glare_elsewhere = sum(c["total"] for c in op13_others)
    print(f"  [inf] OPs the report reports glare-free: ours total={glare_elsewhere} min"
          f" (expect 0; our model has no obstructions)")
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"  [inf] OPs the report reports glare-free: ours total={glare_elsewhere} min"
                f" (expect 0; our model has no obstructions)\n")

    # route per-vertex detail: which route points actually produce glare
    print("\n  Route vertices with glare (minutes, per vertex):")
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write("\n  Route vertices with glare (minutes, per vertex):\n")
    for (a, r), d in details.items():
        if d.empty:
            continue
        c = _count(d)
        line = (f"    {r:<16} G={c['G']:4d} Y={c['Y']:5d} tot={c['total']:5d}")
        print(line)
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    print(f"\n=== RESULT: {PASS} passed, {FAIL} failed "
          f"(elapsed {time.time() - t_start:.0f}s) ===")
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"=== RESULT: {PASS} passed, {FAIL} failed "
                f"(elapsed {time.time() - t_start:.0f}s) ===\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())