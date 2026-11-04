"""Re-render contour figures from existing contour_grid.csv files
Writes PNGs with TRANSPARENT background (facecolor='none').

Usage:
    python scripts/re_render_contour_transparent.py
"""
import os
import sys
import copy
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solar_glare import contour as C
from solar_glare.analysis import load_config, build_site
from solar_glare.contour import GRID_PREFIX


def apply_case_overrides(cfg, case_label):
    """Apply case's azimuth_deg/tilt_deg/env to arrays.

    Mirrors the override order in run_analysis.py -- azimuth_deg and
    tilt_deg on every array, then the optional per-case `env` mapping.
    """
    work = copy.deepcopy(cfg)
    cases_list = work.get("cases") or work.get("site", {}).get("cases", [])
    for case in cases_list:
        if case.get("label") == case_label:
            if "azimuth_deg" in case:
                for arr in work["arrays"]:
                    arr["azimuth_deg"] = float(case["azimuth_deg"])
            if "tilt_deg" in case:
                for arr in work["arrays"]:
                    arr["tilt_deg"] = float(case["tilt_deg"])
            if case.get("env"):
                work.setdefault("environment", {}).update(case["env"])
            break
    return work


def main():
    base_cfg = load_config(os.path.join(ROOT, "config", "site_contour.yaml"))
    root = ROOT

    cases = ["az278_e903_t15", "az278_e903_t20"]

    for case in cases:
        csv_path = os.path.join(ROOT, "output", "contour", case, "results",
                                "contour_grid.csv")
        if not os.path.exists(csv_path):
            print(f"SKIP {case}: missing {os.path.relpath(csv_path, ROOT)}")
            continue

        # Apply case overrides so array orientation matches the case
        cfg = apply_case_overrides(base_cfg, case)
        site, arrays, receptors = build_site(cfg)
        grid, _ = C.grid_from_receptors(receptors, GRID_PREFIX)

        df = pd.read_csv(csv_path)

        summary = df.set_index(["name", "array"])[[
            "minutes_with_glare", "iec_worst_risk",
            "annual_peak_retinal_wcm2", "iec_min_tmax_thermal_s",
            "iec_min_tmax_blue_s", "iec_peak_lr_w_sr_m2",
        ]]
        summary.index.names = ["receptor", "array"]

        fields = C.fields_from_summary(summary, grid, arrays)

        fig_dir = os.path.join(ROOT, "output", "contour", case, "figures_transparent")
        os.makedirs(fig_dir, exist_ok=True)

        written = C.plot_risk_contour_set(site, arrays, grid, fields,
                                         fig_dir, case_label=case,
                                         transparent=True)
        for p in written:
            print(f"  wrote {os.path.relpath(p, ROOT)}")

        html_path = os.path.join(fig_dir, "risk_contours.html")
        C.plot_risk_contours_html(cfg, site, arrays, grid, fields, html_path,
                                  case_label=case)
        print(f"  wrote {os.path.relpath(html_path, ROOT)} ({case})")


if __name__ == "__main__":
    main()
