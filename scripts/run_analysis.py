"""Run the full-year solar glare analysis and generate all report figures.

Usage (from repo root, inside the venv):
    .venv\\Scripts\\python.exe scripts\\run_analysis.py [--case LABEL]

If config/site.yaml has a site.cases list, one full analysis is run per case
(azimuth_deg and/or tilt_deg of every configured array are overridden) and each
case's outputs (figures + results) are written to output/<label>/. Otherwise a
single run goes straight to output/ using each array's configured azimuth_deg
and tilt_deg.

--case LABEL restricts the run to that single case (useful for running the
cases of a design sweep independently); without it, all configured cases run in
one pass.

--brief only trims the CONSOLE output of a 6-case sweep: the per-case table
keeps just the receptors that show glare (the other 63 rows are all zeros) and
the 47-row per-case timing table is left out, because the sweep roll-up printed
at the end already carries the per-case phase timing.  It is applied after all
the computation, so results, CSVs, figures and profile rows are identical.
scripts/stages/step1_analysis.py passes it.

Every run is also profiled (src/solar_glare/profiling.py).  This is opt-in
INFORMATION ONLY: the timer only observes durations, it never feeds the
physics, and the validators run with it disabled so their bit-identity checks
stay meaningful.  Each case writes:

    output/<label>/results/run_profile.csv     per-function timing + provenance
    output/<label>/results/run_spans.csv       per-call timeline (one row per
                                              stage entry/exit: parent, depth,
                                              thread, t0, t1, self time)
    output/<label>/figures/14_run_profile.png  phase / function / parallel view

and a multi-case sweep additionally writes output/run_profile_all_cases.csv.
The run_spans.csv files are what scripts/build_run_timeline.py turns into the
interactive output/run_timeline.html.
Numbers that are already written to another CSV or PNG are deliberately NOT
repeated in the profile; the profile records timing, not results.
"""

import copy
import glob
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solar_glare.analysis import load_config, run_analysis, compute_ephemeris
from solar_glare.profiling import StageTimer, set_timer, PHASES
from solar_glare import profiling as P
from solar_glare import viz, viz3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "site.yaml")
OUT = os.path.join(ROOT, "output")

# Stages paid once for the whole sweep rather than per case (config parsing) get
# their own timer and appear in the roll-up under this pseudo-case label.
ONCE = "__once__"


def run_case(cfg, out_dir, case_label="", profile=True, brief=False):
    """Run one case end to end. Returns the StageTimer so main() can roll the
    per-case profiles up into a single CSV.

    brief=True only shortens what is PRINTED: the per-case table keeps just the
    receptors that actually show glare (the other 63 rows are all zeros) and the
    per-case timing table is skipped, because the sweep roll-up main() prints at
    the end already carries the per-case phase timing.  It is applied strictly
    after all the computation, so every CSV / figure / profile row is identical
    either way.  See scripts/stages/step1_analysis.py, which passes it.
    """
    t0 = time.time()
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)

    # Clear all stale figures before each run so orphaned PNGs never linger.
    for stale in glob.glob(os.path.join(out_dir, "figures", "*.png")):
        os.remove(stale)
    for stale in glob.glob(os.path.join(out_dir, "figures", "*.html")):
        os.remove(stale)

    timer = StageTimer(case_label=case_label, enabled=bool(profile))
    set_timer(timer)
    try:
        times, pos, dni, long_df, summary, details, site, arrays, receptors = run_analysis(
            cfg, progress=True)

        # ---- results CSV (glare rows only; non-glare minutes are omitted) ----
        with timer.stage("write_csvs") as rec:
            n_csv = 0
            long_df.to_csv(os.path.join(out_dir, "results", "glare_events.csv"),
                           index=False)
            n_csv += 1
            summary.to_csv(os.path.join(out_dir, "results", "summary.csv"))
            n_csv += 1
            for (a, r), d in details.items():
                if not d.empty:
                    safe = (a.replace("/", "_").replace(" ", "_") + "__" +
                            r.replace("/", "_").replace(" ", "_")).replace(":", "")
                    d.to_csv(os.path.join(out_dir, "results",
                                          f"glare_hours_{safe}.csv"), index=False)
                    n_csv += 1
            if cfg.get("environment", {}).get("iec62471_enabled"):
                iec_cols = ["iec_worst_risk", "iec_min_tmax_thermal_s",
                            "iec_min_tmax_blue_s", "iec_peak_lr_w_sr_m2"]
                summary[iec_cols].to_csv(
                    os.path.join(out_dir, "results", "iec62471_summary.csv"))
                n_csv += 1
            rec.key(n_files_written=n_csv)

        # ---- solar ----
        # Use cached ephemeris from run_analysis to avoid duplicate computation (~8s saved)
        pos_df = pos

        # ---- figures ----
        # Every figure stage below is nested inside one `draw_figures` block so
        # that block's wall time is the INCLUSIVE figure cost, which is what the
        # phase bar in panel (a) reports.  The individual plot_* stages still
        # keep their own exclusive self times.
        with timer.stage("draw_figures"):
            # ---- site maps ----
            with timer.stage("plot_site_map") as rec:
                viz.plot_site_map(
                    cfg, site, arrays, receptors,
                    os.path.join(out_dir, "figures", "01_site_map_white.png"),
                    transparent=False)
                viz.plot_site_map(
                    cfg, site, arrays, receptors,
                    os.path.join(out_dir, "figures", "01_site_map_trans.png"),
                    transparent=True)
                rec.key(n_figures_written=2)
            with timer.stage("plot_site_map_hazard") as rec:
                viz.plot_site_map_hazard(
                    cfg, site, arrays, receptors, summary,
                    os.path.join(out_dir, "figures", "02_site_map_glare.png"))
                rec.key(n_figures_written=1)

            # ---- fig 03: per-receptor glare-producing sub-panel maps ----
            with timer.stage("plot_group_glare_map") as rec:
                n03 = 0
                for r in receptors:
                    if not any(not d.empty and "group" in d.columns
                               for (a, rr), d in details.items() if rr == r.name):
                        continue
                    safe = r.name.replace("/", "_").replace(" ", "_").replace(":", "")
                    viz.plot_group_glare_map(
                        cfg, site, arrays, receptors, details,
                        os.path.join(out_dir, "figures", f"03_{safe}_glare_map.png"),
                        receptor=r.name)
                    n03 += 1
                rec.key(n_figures_written=n03)

            with timer.stage("plot_site_map_folium") as rec:
                try:
                    viz.plot_site_map_folium(
                        cfg, site, arrays, receptors,
                        os.path.join(out_dir, "figures", "01_site_map.html"))
                    rec.key(n_figures_written=1)
                except Exception as exc:  # folium needs internet for tiles
                    print(f"[warn] folium map skipped: {exc}")

            # ---- sun-path figure ----
            with timer.stage("plot_sun_path") as rec:
                viz.plot_sun_path(
                    pos_df, "Annual sun path - Map Ta Phut (13 deg N)",
                    os.path.join(out_dir, "figures", "04_sun_path.png"),
                    lat=site["reference"]["lat"], lon=site["reference"]["lon"],
                    altitude=site["reference"].get("height_m", 0.0))
                rec.key(n_figures_written=1)

            # ---- per-pair glare heatmaps ----
            with timer.stage("plot_glare_heatmap") as rec:
                n05 = 0
                for (a, r), d in details.items():
                    if d.empty:
                        continue
                    safe = (a.split("(")[0].strip() + "_" + r.split(" ")[0]).replace(" ", "_")
                    viz.plot_glare_heatmap(
                        d, f"Glare occurrence - {a} as seen from {r}",
                        os.path.join(out_dir, "figures", f"05_heatmap_{safe}.png"))
                    n05 += 1
                rec.key(n_figures_written=n05)

            with timer.stage("plot_daily_duration") as rec:
                n06 = 0
                for (a, r), d in details.items():
                    if d.empty:
                        continue
                    safe = (a.split("(")[0].strip() + "_" + r.split(" ")[0]).replace(" ", "_")
                    viz.plot_daily_duration(
                        d, f"Daily glare duration - {a} as seen from {r}",
                        os.path.join(out_dir, "figures", f"06_daily_duration_{safe}.png"))
                    n06 += 1
                rec.key(n_figures_written=n06)

            # ---- summary + ocular + sun-position charts ----
            iec_enabled = bool(cfg.get("environment", {}).get("iec62471_enabled"))
            with timer.stage("plot_summary") as rec:
                viz.plot_summary(summary, details,
                                 os.path.join(out_dir, "figures", "07_hazard_summary.png"),
                                 iec_enabled=iec_enabled)
                rec.key(n_figures_written=1)
            with timer.stage("plot_ocular_chart") as rec:
                viz.plot_ocular_chart(
                    details, os.path.join(out_dir, "figures", "08_ocular_chart.png"),
                    legend_path=os.path.join(out_dir, "figures",
                                             "08_ocular_chart_legend.png"))
                rec.key(n_figures_written=2)
            with timer.stage("plot_sun_chart") as rec:
                viz.plot_sun_chart(
                    details, os.path.join(out_dir, "figures", "09_glare_by_sunpos.png"),
                    iec_enabled=iec_enabled)
                rec.key(n_figures_written=1)

            # ---- IEC 62471 secondary assessment (when enabled in config) ----
            if cfg.get("environment", {}).get("iec62471_enabled"):
                with timer.stage("plot_iec62471_zones") as rec:
                    viz.plot_iec62471_zones(
                        details, os.path.join(out_dir, "figures", "10_iec62471_zones.png"))
                    rec.key(n_figures_written=1)
                with timer.stage("plot_iec62471_tmax") as rec:
                    viz.plot_iec62471_tmax(
                        details,
                        os.path.join(out_dir, "figures", "11_iec62471_tmax_thermal.png"),
                        os.path.join(out_dir, "figures", "11_iec62471_tmax_blue.png"))
                    rec.key(n_figures_written=2)
                with timer.stage("plot_iec62471_weights") as rec:
                    viz.plot_iec62471_weights(
                        os.path.join(out_dir, "figures",
                                     "12_iec62471_weighting_functions.png"))
                    rec.key(n_figures_written=1)
                with timer.stage("plot_iec62471_spectrum_weighted") as rec:
                    viz.plot_iec62471_spectrum_weighted(
                        os.path.join(out_dir, "figures",
                                     "12b_iec62471_spectrum_weighted.png"))
                    rec.key(n_figures_written=1)
            # ---- 3D worst-case geometry (one per receptor) ----
            with timer.stage("plot_3d_geometry") as rec:
                n13 = 0
                for r in receptors:
                    worst = None
                    for (a, rr), d in details.items():
                        if rr != r.name or d.empty:
                            continue
                        if worst is None or d["retinal_wcm2"].max() > worst["detail"]["retinal_wcm2"].max():
                            worst = {"array": a, "detail": d}
                    if worst is None:
                        continue
                    arr = next(a for a in arrays if a.name == worst["array"])
                    peak = worst["detail"].loc[worst["detail"]["retinal_wcm2"].idxmax()]
                    sum_row = summary.loc[(r.name, worst["array"])]
                    viz3d.plot_glare_geometry(
                        arr, r, peak["azimuth_deg"], peak["elevation_deg"],
                        os.path.join(out_dir, "figures",
                                     f"13_3d_geometry_{r.name.split()[0]}.png"),
                        title=f"Worst-case glare geometry - {r.name}",
                        all_arrays=arrays,
                        occurrence_time=peak["time"].strftime("%Y-%m-%d %H:%M"),
                        g_min=int(sum_row["G"]), y_min=int(sum_row["Y"]),
                        r_min=int(sum_row["R"]),
                    )
                    n13 += 1
                rec.key(n_figures_written=n13)

        print(f"  ANALYSIS COMPLETE ({time.time() - t0:.1f}s elapsed)")
        if brief:
            hot = summary[summary["minutes_with_glare"] > 0]
            print(hot.to_string() if not hot.empty
                  else "  (no glare at any receptor)")
        else:
            print(summary.to_string())
    finally:
        set_timer(None)

    # ---- figure 14 + the profile CSV itself ----
    # These reuse the `draw_figures` / `write_csvs` phase roots, so their time
    # still lands in the right phase of panel (a).  A profiler cannot measure
    # the write of its own log, so the final `write_csvs` row is not in the CSV.
    if profile:
        with timer.stage("draw_figures") as rec:
            viz.plot_run_profile(
                timer, os.path.join(out_dir, "figures", "14_run_profile.png"),
                case_label=case_label)
            rec.key(n_figures_written=1)
        with timer.stage("write_csvs") as rec:
            prof = os.path.join(out_dir, "results", "run_profile.csv")
            timer.to_frame().assign(case=case_label).to_csv(prof, index=False)
            # The event timeline behind output/run_timeline.html.  Written
            # AFTER the aggregate profile on purpose: the aggregate is the
            # authoritative per-stage view, and the span rows are raw
            # observations (thread/parent/depth) that the HTML builder
            # re-aggregates for its inventory table.
            spans = os.path.join(out_dir, "results", "run_spans.csv")
            sdf = timer.call_span_frame()
            if not sdf.empty:
                labels = timer.thread_labels()
                sdf = sdf.assign(phase=[P.STAGE_SPECS[st]["phase"] for st in sdf.stage],
                                 thread_label=[labels[i] for i in sdf.thread],
                                 case=case_label)
                # 6 decimals = 1 microsecond, plenty for a wall-clock timeline
                # and ~4x smaller than pandas' default repr.
                sdf.to_csv(spans, index=False, float_format="%.6f")
            rec.key(n_files_written=1)
        print(f"  run profile -> {os.path.relpath(prof, ROOT)}")
        if not sdf.empty:
            print(f"  run spans   -> {os.path.relpath(spans, ROOT)} "
                  f"({len(sdf)} spans, {sdf.thread.nunique()} thread objects -> "
                  f"{timer.n_thread_lanes()} worker lanes + main, "
                  f"depth<={int(sdf.depth.max())})")
        if not brief:
            print(timer.format_table())
    # Freeze the case total here; otherwise it would keep accumulating through
    # the remaining cases of the sweep.
    timer.stop()
    return timer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", metavar="LABEL", default=None,
                        help="run only this single case from site.cases")
    parser.add_argument("--brief", action="store_true",
                        help="print only the receptors that show glare, and skip "
                             "the per-case timing table (results unchanged)")
    args = parser.parse_args()

    # load_config runs once for the whole sweep, before any case timer exists,
    # so it gets its own timer and is folded into the roll-up as `__once__`.
    once = StageTimer(case_label=ONCE, enabled=True)
    set_timer(once)
    try:
        cfg = load_config(CFG)
    finally:
        set_timer(None)
        once.stop()
    cases = cfg.get("site", {}).get("cases") or []

    if args.case:
        wanted = [c for c in cases if c.get("label") == args.case]
        if not wanted:
            known = ", ".join(c.get("label") for c in cases) or "(none)"
            raise SystemExit(f"case {args.case!r} not in site.cases "
                             f"(known: {known})")
        cases = wanted

    timers = [(ONCE, once)]
    if not cases:
        timers.append(("", run_case(cfg, OUT, brief=args.brief)))
    else:
        for case in cases:
            label = case.get("label")
            if not label:
                label = f"az{float(case['azimuth_deg']):g}"
            work = copy.deepcopy(cfg)
            az = None
            if "azimuth_deg" in case:
                az = float(case["azimuth_deg"])
                for arr in work["arrays"]:
                    arr["azimuth_deg"] = az
            tilt = None
            if "tilt_deg" in case:
                tilt = float(case["tilt_deg"])
                for arr in work["arrays"]:
                    arr["tilt_deg"] = tilt
            az_txt = f"{az:g}" if az is not None else "default"
            tilt_txt = f"{tilt:g}" if tilt is not None else "default"
            print(f"=== case {label}: azimuth_deg = {az_txt} "
                  f"tilt_deg = {tilt_txt} ===")
            # Optional per-case environment overrides (e.g. a different
            # reflectivity_mode for a method-comparison run); absent = unchanged.
            env_over = case.get("env")
            if env_over:
                work.setdefault("environment", {}).update(env_over)
                print(f"    environment overrides: {env_over}")
            timers.append((label, run_case(work, os.path.join(OUT, label),
                                           case_label=label, brief=args.brief)))
            print()

    # ---- roll-up across the sweep ----
    import pandas as pd
    frames = [t.to_frame().assign(case=lbl) for lbl, t in timers
              if not t.to_frame().empty]
    if frames:
        allf = pd.concat(frames, ignore_index=True)
        rollup = os.path.join(OUT, "run_profile_all_cases.csv")
        allf.to_csv(rollup, index=False)
        print("=== sweep roll-up: wall time per phase per case ===")
        rows = []
        for lbl, t in timers:
            pf = t.phase_frame().set_index("phase")["wall_s"]
            rec = {"case": lbl or "(default)", "total_wall_s": round(t.total_wall_s, 1)}
            for p in PHASES:
                rec[p] = round(float(pf.get(p, 0.0)), 2)
            rows.append(rec)
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
        tot = table.drop(columns=["case", "total_wall_s"]).sum()
        print("\n" + tot.to_string())
        print(f"\nTOTAL sweep wall time: {table['total_wall_s'].sum():.1f} s "
              f"over {len(table)} timer(s)")
        print(f"profile roll-up -> {os.path.relpath(rollup, ROOT)}")


if __name__ == "__main__":
    main()
