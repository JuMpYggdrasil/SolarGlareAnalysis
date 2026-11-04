"""Run the pipeline over the dense GROUND-LEVEL contour grid and draw isorisk maps.

Usage (from repo root, inside the venv):
    .venv\\Scripts\\python.exe scripts\\run_contour.py [--case LABEL] [--all]
                 [--grid-only] [--no-html]

This is deliberately NOT `run_analysis.py`. Two reasons:

* it reads `config/site_contour.yaml`, not `config/site.yaml`, so the committed
  receptor study and its figures are never touched; and
* it keeps only ONE output per case -- `contour_grid.csv`, `risk_contours.png`
  and `risk_contours.html` -- instead of the 6 receptors x per-pair detail CSVs
  that would otherwise pile up for 703 extra receptors.

The physics is untouched. A contour grid point is an ordinary receptor, so
`analysis.run_analysis` classifies it with the same polygon detector, SGHAT
ocular hazard and IEC 62471 logic the committed cases use; this script only
reshapes the returned `summary` into a 2-D field and plots it. Profiling stays
OFF (`StageTimer(enabled=False)` default) so nothing perturbs the computation.

Because the grid points sit at 0 m AGL rather than the study's 40 m, the RG
zones reported here are a different question from the 65-point receptor study:
they answer "where on the ground would a person be standing in a risk group",
not "what did we measure at an observation point".

`--case LABEL` runs one configured case; `--all` is the default and runs every
entry of `site.cases` in `config/site_contour.yaml`.
"""

import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solar_glare.analysis import load_config, run_analysis  # noqa: E402
from solar_glare.contour import (  # noqa: E402
    GRID_PREFIX,
    VAL_PREFIX,
    contour_grid_csv,
    fields_from_summary,
    grid_from_receptors,
    peak_provenance,
    plot_risk_contour_set,
    plot_risk_contours_html,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "site_contour.yaml")
OUT = os.path.join(ROOT, "output", "contour")


def apply_case(cfg, case):
    """Return a deep copy of `cfg` with the case's array/env overrides applied.

    Mirrors the override order in `run_analysis.py:main` -- azimuth_deg and
    tilt_deg on every array, then the optional per-case `env` mapping -- so a
    contour case means exactly what the committed case of the same label means.
    """
    work = copy.deepcopy(cfg)
    if "azimuth_deg" in case:
        for arr in work["arrays"]:
            arr["azimuth_deg"] = float(case["azimuth_deg"])
    if "tilt_deg" in case:
        for arr in work["arrays"]:
            arr["tilt_deg"] = float(case["tilt_deg"])
    if case.get("env"):
        work.setdefault("environment", {}).update(case["env"])
    return work


_EXACT_COLS = ["N", "G", "Y", "R", "minutes_with_glare", "iec_worst_risk"]
_TOL_COLS = ["annual_peak_retinal_wcm2", "iec_min_tmax_thermal_s",
             "iec_min_tmax_blue_s", "iec_peak_lr_w_sr_m2"]


def validate_against_committed(summary, case_label, root=ROOT):
    """Check the `v_` validation receptors against the committed 65-point run.

    The contour config keeps a renamed copy of every study receptor
    (`v_<name>`, original 40 m AGL). Those points go through exactly the same
    pipeline and array settings as the committed case, so their summary rows
    must match `output/<label>/results/summary.csv`: G/Y/R, minutes_with_glare
    and iec_worst_risk exactly, and the peak/t_max values to 1e-9 relative. That
    makes this run self-checking -- if the contour config ever drifted from the
    committed study, the figures would otherwise be silently describing a
    different model.

    Returns (n_compared, [mismatch strings], [note strings]). A case with no
    committed reference CSV is reported as SKIPPED, never as a pass.
    """
    import numpy as np
    import pandas as pd

    notes, bad = [], []
    ref_path = os.path.join(root, "output", case_label, "results", "summary.csv")
    if not os.path.exists(ref_path):
        return 0, [], [f"no committed reference at "
                       f"{os.path.relpath(ref_path, root)} - SKIPPED"]

    ref = pd.read_csv(ref_path).set_index(["receptor", "array"])
    val = summary[summary.index.get_level_values("receptor")
                  .str.startswith(VAL_PREFIX)].copy()
    if val.empty:
        return 0, [], ["no v_* receptors in this run (--grid-only): "
                       "validation skipped"]
    # Rename ONLY the receptor level. Assigning a flat Index here would drop the
    # array level, and Index.intersection(MultiIndex) on strings vs tuples is
    # silently EMPTY -- which reported "matched 0/65 receptors" instead of
    # raising, i.e. a validation that could never pass.
    val.index = pd.MultiIndex.from_arrays(
        [[s[len(VAL_PREFIX):] for s in val.index.get_level_values("receptor")],
         val.index.get_level_values("array")],
        names=["receptor", "array"])
    common = val.index.intersection(ref.index)
    if len(common) != len(ref.index):
        notes.append(f"matched only {len(common)}/{len(ref.index)} committed "
                     f"receptors (renaming mismatch?)")

    for receptor in common:
        a, b = val.loc[receptor], ref.loc[receptor]
        for c in _EXACT_COLS:
            if c in b and not _same(a.get(c), b.get(c)):
                bad.append(f"{receptor}.{c}: {a.get(c)!r} != {b.get(c)!r}")
        for c in _TOL_COLS:
            if c in b and not _close(a.get(c), b.get(c)):
                bad.append(f"{receptor}.{c}: {a.get(c)} != {b.get(c)}")
    return len(common), bad, notes


def _same(a, b):
    """Exact equality for the integer / label columns."""
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    if isinstance(a, (int, float, np.integer, np.floating)):
        return float(a) == float(b)
    return str(a).strip() == str(b).strip()


def _close(a, b, rtol=1e-9):
    """Relative comparison for the peak / t_max columns."""
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    return abs(float(a) - float(b)) <= rtol * max(1.0, abs(float(b)))


def _missing(v):
    if v is None:
        return True
    try:
        import pandas as pd
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _roundtrip_report(df, rtol=1e-9):
    """Verify the two published identities on every non-glare grid cell.

        E_c = DNI/1e4 * rho * tau          (Ho 2011 Eq. 1)
        E_r = E_c * k_f / x**2              (Ho 2011 Eq. 2)

    k_f = (pupil/focal)^2 * tau_eye, taken from `hazard` rather than retyped, so
    this cannot drift from the module that produced the numbers. Both sides are
    computed only from stored columns, so this is a genuine independent check
    and not a tautology. Returns "" on success, else a short failure report.
    """
    from solar_glare.hazard import (EYE_FOCAL_LENGTH, OCULAR_TRANSMISSION,
                                    PUPIL_DIAMETER)
    k_f = PUPIL_DIAMETER ** 2 / EYE_FOCAL_LENGTH ** 2 * OCULAR_TRANSMISSION

    g = df.dropna(subset=["annual_peak_retinal_wcm2"])
    if g.empty:
        return "no glare cells to check"
    ec = (g["peak_dni_wm2"] / 1e4 * g["peak_reflectivity_used"]
          * g["peak_atmos_transmittance"])
    er = g["peak_corneal_wcm2"] * k_f / g["peak_subtended_rad"] ** 2
    ec_bad = ~np.isclose(ec, g["peak_corneal_wcm2"], rtol=rtol, atol=0.0)
    er_bad = ~np.isclose(er, g["annual_peak_retinal_wcm2"], rtol=rtol, atol=0.0)
    if not (ec_bad.any() or er_bad.any()):
        return ""
    return (f"{int(ec_bad.sum())}/{len(g)} E_c and {int(er_bad.sum())}/{len(g)} "
            f"E_r mismatches (worst cell {g.name[er_bad | ec_bad][:1].tolist()})")


def run_case(work, label, base_out_dir, html=True, hour_range=None, root=ROOT):
    """Run one contour case end to end and write its three artefacts."""
    t0 = time.time()

    # run_analysis returns (times, pos, dni, long_df, summary, details, site, arrays, receptors); the contour needs the last four. long_df and details are
    # discarded -- for 703 receptors they are large and unused here.
    _times, _pos, _dni, _long, summary, details, site, arrays, receptors = run_analysis(work, hour_range=hour_range, progress=True)
    grid, _others = grid_from_receptors(receptors, GRID_PREFIX)

    # Determine height-based subdirectory from grid receptors (all share same height_agl_m)
    height_dir = f"height_{int(grid['height_agl_m'])}"
    out_dir = os.path.join(base_out_dir, height_dir, label)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    # Clear this case's figures first. The panel split replaced the single
    # combined `risk_contours.png`, and a stale copy of that file sitting beside
    # the ten new panels reads as current output. Only THIS case's directory is
    # touched, so cases not in the config keep the figures they already have.
    fig_dir = os.path.join(out_dir, "figures")
    for stale in os.listdir(fig_dir):
        os.remove(os.path.join(fig_dir, stale))
    fields = fields_from_summary(summary, grid, arrays)
    prov = peak_provenance(details, arrays, grid)

    csv_path = os.path.join(out_dir, "results", "contour_grid.csv")
    df = contour_grid_csv(grid, fields, arrays, csv_path, provenance=prov)

    # Round-trip the published identities on the peak-minute row of every
    # dangerous cell: E_c = DNI/1e4 * rho * tau and E_r = E_c * k_f / x**2.
    # This is the project's own known-answer check (validate_charts [A]) applied
    # to the contour, so the map cannot quietly disagree with the pipeline.
    # Reported either way: a check that prints only on failure leaves a pass
    # looking exactly like a check that never ran.
    bad = _roundtrip_report(df)
    n_checked = int(df["annual_peak_retinal_wcm2"].notna().sum())
    print(f"    E_c/E_r round-trip: {bad or 'ok'}"
          f" ({n_checked} glare cells, rtol 1e-9)")

    # Generate both opaque and transparent figures in a single pass
    fig_dir = os.path.join(out_dir, "figures")
    # Remove old figures_transparent if exists
    import shutil
    trans_dir_old = os.path.join(out_dir, "figures_transparent")
    if os.path.isdir(trans_dir_old):
        shutil.rmtree(trans_dir_old, ignore_errors=True)
    os.makedirs(fig_dir, exist_ok=True)
    for png_path in plot_risk_contour_set(site, arrays, grid, fields,
                                          fig_dir,
                                          case_label=label,
                                          transparent=True):
        print(f"  wrote {os.path.relpath(png_path, root)}")

    html_path = os.path.join(fig_dir, "risk_contours.html")
    if html:
        plot_risk_contours_html(work, site, arrays, grid, fields, html_path,
                                case_label=label)
        print(f"  wrote {os.path.relpath(html_path, root)}")

    n_val = len([r for r in receptors if r.name.startswith(VAL_PREFIX)])
    print(f"  grid {grid['nrow']}x{grid['ncol']} = {len(grid['names'])} pts "
          f"(0 m AGL) + {n_val} validation pts")
    # fields["rg_label"] is a numpy object array, not a Series, so count with
    # np.unique rather than pandas .value_counts().
    seen = dict(zip(*np.unique(fields["rg_label"], return_counts=True)))
    for name in ("Exempt", "RG1", "RG2", "RG3"):
        print(f"    {name:<7} {int(seen.get(name, 0)):>5} pts")
    print(f"    glare minutes on grid: {int(fields['minutes'].sum())}")

    # Auto-skip validation when hour_range filters time range (results not comparable to full-year committed data)
    if hour_range:
        print(f"    validation: skipped (hour_range={hour_range} filters time range)")
        n_cmp, bad, notes = 0, [], [f"hour_range={hour_range}: validation skipped"]
    else:
        n_cmp, bad, notes = validate_against_committed(summary, label, root)
    for n in notes:
        print(f"    validation: {n}")
    if n_cmp:
        print(f"    validation vs committed {label}: {n_cmp} receptors, "
              f"{len(bad)} mismatched")
    for m in bad:
        print(f"    MISMATCH {m}")
    print(f"  wrote {os.path.relpath(csv_path, root)}")
    print(f"  wrote {os.path.relpath(png_path, root)}")
    if html:
        print(f"  wrote {os.path.relpath(html_path, root)}")
    print(f"  ({time.time() - t0:.0f} s)")
    return dict(n_cmp=n_cmp, n_bad=len(bad))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--case", help="run one configured case label")
    p.add_argument("--all", action="store_true",
                   help="run every case (default)")
    p.add_argument("--grid-only", action="store_true",
                   help="run only the 703 grid receptors, skip the v_* "
                        "validation points (faster, no self-check)")
    p.add_argument("--no-html", action="store_true",
                   help="skip the interactive folium map")
    p.add_argument("--hour-range", nargs=2, type=int, metavar=("START", "END"),
                   help="Limit analysis to hour range inclusive (0-23), e.g. --hour-range 7 8 for 07:00-08:59")
    args = p.parse_args(argv)

    cfg = load_config(CFG)

    # Determine hour_range: CLI > config > None
    hour_range = None
    if args.hour_range:
        hour_range = tuple(args.hour_range)
    else:
        hr = cfg.get("site", {}).get("hour_range")
        if hr:
            hour_range = tuple(hr)

    cases = (cfg.get("site") or {}).get("cases") or []
    if args.case:
        cases = [c for c in cases if c.get("label") == args.case]
        if not cases:
            known = ", ".join(c.get("label", "?") for c in load_config(CFG)
                              .get("site", {}).get("cases", []))
            raise SystemExit(f"case {args.case!r} not in "
                             f"config/site_contour.yaml site.cases (known: {known})")

    if args.grid_only:
        # Drop the renamed study receptors so only the 703 grid points run.
        # NOTE: `receptors` is a TOP-LEVEL config key (analysis.build_site reads
        # cfg["receptors"], not cfg["site"]["receptors"]).
        cfg["receptors"] = [r for r in cfg["receptors"]
                            if r["name"].startswith(GRID_PREFIX)]

    if not cases:
        raise SystemExit("config/site_contour.yaml has no site.cases; contour "
                         "runs are driven by the case sweep")

    os.makedirs(OUT, exist_ok=True)
    total_bad = 0
    for case in cases:
        label = case.get("label") or f"az{float(case['azimuth_deg']):g}"
        az = case.get("azimuth_deg")
        tilt = case.get("tilt_deg")
        print(f"=== contour case {label}: azimuth_deg = "
              f"{f'{float(az):g}' if az is not None else 'default'}, "
              f"tilt_deg = "
              f"{f'{float(tilt):g}' if tilt is not None else 'default'} ===")
        res = run_case(apply_case(cfg, case), label, OUT,
                       html=not args.no_html, hour_range=hour_range)
        total_bad += res["n_bad"]
        print()
    if total_bad:
        print(f"WARNING: {total_bad} validation mismatches against the "
              f"committed runs -- see the MISMATCH lines above")
    return 0


if __name__ == "__main__":
    sys.exit(main())