"""Validation of the ASTM E903-20-style solar-weighted reflectance mode.

Reference: ASTM E903-20 "Solar Absorptance, Reflectance, and Transmittance of
Materials Using Integrating Spheres" (abstract/scope only; full text is
paywalled) + the free equivalent SolarPACES Reflectance Guideline V3.1
(references/_SolarPACES_Reflectance_Guideline_V3.1.txt).

This validates the *model* implementation of E903-20's solar-weighting
procedure in `solar_glare/hazard.py::solar_weighted_reflectivity`
(Fresnel spectral model of the cover glass weighted by the IEC 60904-3:2019
direct-normal spectrum over [320, 2500] nm) and compares it against the
SGHAT v6 Table 1 empirical AOI fits (kept in hazard.py as a reference
comparison; they are not a selectable pipeline mode).

Run from repo root (inside the venv):
    .venv\\Scripts\\python.exe scripts\\validate_e903.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solar_glare.hazard import (
    aoi_reflectivity,
    solar_weighted_reflectivity,
    glass_refractive_index,
    PUPIL_DIAMETER,
    EYE_FOCAL_LENGTH,
    OCULAR_TRANSMISSION,
    _E903_GLASS_TYPES,
    _E903_WL_MIN,
    _E903_WL_MAX,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "output", "validation")
OUT_PATH = os.path.join(OUT_DIR, "e903.txt")
FIG_PATH = os.path.join(OUT_DIR, "e903_refl_compare.png")

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def line(txt=""):
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(txt + "\n")


# ---------------------------------------------------------------------------
# [A] Known-answer checks on the E903-style model
# ---------------------------------------------------------------------------
def checks_model():
    print("\n[A] E903-20-style model known-answers")
    # cover-glass dispersion
    n550 = float(glass_refractive_index(550.0))
    n2500 = float(glass_refractive_index(2500.0))
    check("n(550) ~ 1.52 (low-iron cover glass)",
          abs(n550 - 1.5185) < 1e-3, f"n={n550:.4f}")
    check("dispersion n(320) > n(2500)",
          float(glass_refractive_index(320.0)) > n2500,
          f"{float(glass_refractive_index(320.0)):.4f} > {n2500:.4f}")

    # single-interface normal-incidence near the SGHAT smooth 4.41% (our ~4.18%)
    r0 = float(solar_weighted_reflectivity(0.0, glass="smooth"))
    check("R(0) smooth ~= SGHAT smooth fit (0.0441)",
          abs(r0 - 0.0441) < 0.02, f"R(0)={r0:.4f}")
    # grazing Fresnel saturates high (no legacy 0.08 clip of SGHAT)
    r89 = float(solar_weighted_reflectivity(89.0, glass="smooth"))
    r90 = float(solar_weighted_reflectivity(90.0, glass="smooth"))
    check("R(89) smooth > 0.8 (physical Fresnel, not clipped at 0.08)",
          r89 > 0.8, f"R(89)={r89:.4f}")
    check("AOI >= 89 saturates at R(89)",
          abs(r90 - r89) < 1e-9, f"R(90)={r90:.4f}")
    # ARC lowers normal incidence vs bare glass
    a0 = float(solar_weighted_reflectivity(0.0, glass="smooth_arc"))
    a80 = float(solar_weighted_reflectivity(80.0, glass="smooth_arc"))
    check("R(0) smooth_arc < R(0) smooth (ARC antireflects)",
          a0 < r0, f"{a0:.4f} < {r0:.4f}")
    check("R(80) smooth_arc > R(0) smooth_arc (grazing rises)",
          a80 > a0, f"{a80:.4f} > {a0:.4f}")

    # range / vectorisation / scalar alias per glass type
    for g in sorted(_E903_GLASS_TYPES):
        xs = np.linspace(0.0, 90.0, 181)
        v = solar_weighted_reflectivity(xs, glass=g)
        check(f"{g}: vector shape + [0,1] range",
              v.shape == xs.shape and float(v.min()) >= 0.0 and
              float(v.max()) <= 1.0,
              f"min={float(v.min()):.4f} max={float(v.max()):.4f}")
        sc = solar_weighted_reflectivity(40.0, glass=g)
        check(f"{g}: scalar alias returns float",
              isinstance(sc, float) and 0.0 <= sc <= 1.0, f"R(40)={sc:.5f}")
    try:
        solar_weighted_reflectivity(10.0, glass="light_textured")
        check("textured glass rejected (no physical n())", False)
    except ValueError:
        check("textured glass rejected (no physical n())", True)

    # spectral window per SolarPACES V3.1 4.4
    check("solar-weighting window = [320, 2500] nm",
          _E903_WL_MIN == 320.0 and _E903_WL_MAX == 2500.0,
          f"[{_E903_WL_MIN:g}, {_E903_WL_MAX:g}]")


# ---------------------------------------------------------------------------
# [B] Curve comparison vs SGHAT v6 Table 1 fits
# ---------------------------------------------------------------------------
def curve_table():
    print("\n[B] R(AOI) table: SGHAT v6 Table 1 vs E903-20-style solar-weighted")
    aois = np.arange(0.0, 91.0, 5.0)
    rows = [("AOI_deg", "SGHAT_smooth", "SGHAT_smooth_arc",
             "E903_smooth", "E903_smooth_arc")]
    for a in aois:
        rows.append((f"{a:g}",
                     f"{float(aoi_reflectivity(a, glass='smooth')):.4f}",
                     f"{float(aoi_reflectivity(a, glass='smooth_arc')):.4f}",
                     f"{float(solar_weighted_reflectivity(a, glass='smooth')):.4f}",
                     f"{float(solar_weighted_reflectivity(a, glass='smooth_arc')):.4f}"))
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    line("\nR(AOI) comparison table (solar-weighted reflectance; E903 model "
         "over IEC 60904-3:2019 Table 1 direct DNI, [320,2500] nm):")
    line("  " + "  ".join(h.ljust(w) for h, w in zip(rows[0], widths)))
    for r in rows[1:]:
        line("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
    for r in rows[1:]:
        print("  " + " | ".join(c.ljust(w) for c, w in zip(r, widths)))

    # ratio in the glare AOI band (78-82 deg) where this site's hits sit
    band = np.linspace(78.0, 82.0, 21)
    e903_arc = np.array([float(solar_weighted_reflectivity(a, glass="smooth_arc"))
                         for a in band])
    sghat_arc = np.array([float(aoi_reflectivity(a, glass="smooth_arc"))
                          for a in band])
    rat = e903_arc / sghat_arc
    check("E903 smooth_arc: R vs SGHAT smooth_arc ratio in 78-82 deg band "
          "is 1.0-1.8x (Fresnel binds higher at grazing)",
          1.0 <= float(rat.mean()) <= 1.8,
          f"mean ratio={float(rat.mean()):.2f} (min {float(rat.min()):.2f}, "
          f"max {float(rat.max()):.2f})")


def make_fig():
    print("\n[C] Writing comparison figure output/validation/e903_refl_compare.png")
    aoi = np.linspace(0.0, 90.0, 901)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(aoi, aoi_reflectivity(aoi, glass="smooth"),
            color="k", ls="-", lw=1.6, label="SGHAT v6 Table 1 - smooth")
    ax.plot(aoi, aoi_reflectivity(aoi, glass="smooth_arc"),
            color="k", ls="--", lw=1.6, label="SGHAT v6 Table 1 - smooth_arc")
    ax.plot(aoi, solar_weighted_reflectivity(aoi, glass="smooth"),
            color="tab:red", ls="-", lw=1.8,
            label="E903-20-style Fresnel - bare glass")
    ax.plot(aoi, solar_weighted_reflectivity(aoi, glass="smooth_arc"),
            color="tab:red", ls="--", lw=1.8,
            label="E903-20-style Fresnel - quarter-wave ARC")
    ax.axvspan(78.0, 82.0, color="tab:orange", alpha=0.15,
               label="glare AOI band on this site (78-82 deg)")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Angle of incidence (deg)")
    ax.set_ylabel("Solar-weighted reflectance R")
    ax.set_title("Panel reflectivity models: SGHAT v6 Table 1 vs "
                 "E903-20-style solar-weighted Fresnel")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=120)
    plt.close(fig)
    line(f"\nComparison figure: output/validation/e903_refl_compare.png")


# ---------------------------------------------------------------------------
# [D] End-to-end self-consistency on the committed 6-case tilt-sweep output
#     (az098_e903_t10/15/20 + az278_e903_t10/15/20 = solar_weighted rho x
#     Beer-Lambert tau, tilt design sweep 10/15/20 deg). Requires
#     run_analysis.py. The former baseline-vs-combined comparison was removed
#     on 2026-09-25 together with the az098/az278 no-extinction baselines; this
#     block verifies the remaining runs against the pipeline arithmetic
#     themselves (every per-minute operand is stored in the detail CSVs, so the
#     round trips are exact).
# ---------------------------------------------------------------------------
# Geometry is attenuation- and tilt-agnostic in the self-consistency math: hit
# minutes / G-Y-R come from the run, only the irradiance scale is re-derived
# from E903 reflectance x Beer-Lambert transmittance per minute.
_TARGETS = {
    # az098 (panel faces ENE ~98 deg): obs_ptt + west1_route_11, all Yellow.
    ("az098_e903_t10", "obs_ptt"):        dict(g=0, y=489, r=0),
    ("az098_e903_t10", "west1_route_11"): dict(g=0, y=351, r=0),
    ("az098_e903_t15", "obs_ptt"):        dict(g=0, y=466, r=0),
    ("az098_e903_t15", "west1_route_11"): dict(g=0, y=336, r=0),
    ("az098_e903_t20", "obs_ptt"):        dict(g=0, y=470, r=0),
    ("az098_e903_t20", "west1_route_11"): dict(g=0, y=327, r=0),
    # az278 (panel faces W ~278 deg): west far-field route points, Green-only
    # (tau cuts the power below the afterimage threshold). t20 = no glare at all.
    ("az278_e903_t10", "west1_route_1"):  dict(g=152, y=0, r=0),
    ("az278_e903_t10", "west1_route_2"):  dict(g=0, y=104, r=0),
    ("az278_e903_t15", "west1_route_1"):  dict(g=77, y=0, r=0),
}


def case_selfcheck():
    print("\n[D] End-to-end self-consistency of the committed 6-case tilt sweep "
          "(solar_weighted x beer_lambert); requires run_analysis.py.")

    kf = (PUPIL_DIAMETER ** 2) / (EYE_FOCAL_LENGTH ** 2) * OCULAR_TRANSMISSION
    tau_means = {}
    for (label, recp), expect in _TARGETS.items():
        name = f"{label}/{recp}"
        d = _detail(label, recp)
        path = os.path.join(REPO, "output", label, "results", "summary.csv")
        if d is None or not os.path.exists(path):
            print(f"  [skip] {label}/{recp}: case output missing "
                  "(run run_analysis.py first)")
            continue
        df = pd.read_csv(path)
        row = df[(df["receptor"] == recp) & (df["array"] == "PV-A")]
        if row.empty:
            print(f"  [skip] {label}/{recp}: no PV-A row in summary.csv")
            continue
        row = row.iloc[0]
        g, y, r = int(row["G"]), int(row["Y"]), int(row["R"])
        m = int(row["minutes_with_glare"])
        peak = float(row["annual_peak_retinal_wcm2"])

        check(f"{name}: G/Y/R counts = expected "
              f"({expect['g']}/{expect['y']}/{expect['r']})",
              (g, y, r) == (expect["g"], expect["y"], expect["r"]),
              f"{g}/{y}/{r}")
        check(f"{name}: G+Y+R == minutes_with_glare", g + y + r == m,
              f"{g + y + r} vs {m}")
        check(f"{name}: summary peak == max detail retinal",
              abs(peak - float(d["retinal_wcm2"].max())) < 1e-9,
              f"{peak:.4f}")

        # E_c = DNI/1e4 * rho * tau (both the E903 rho and the Beer-Lambert
        # transmittance are stored per minute -> exact round trip)
        pred = d["dni_wm2"] / 1e4 * d["reflectivity_used"] \
            * d["atmos_transmittance"]
        rel = (np.abs(pred - d["corneal_wcm2"])
               / np.maximum(d["corneal_wcm2"], 1e-12)).max()
        check(f"{name}: E_c == DNI/1e4 * rho * tau per minute (n={len(d)})",
              rel < 1e-9, f"max rel err {rel:.2e}")

        # E_r = E_c * (d_p/f)^2 / x^2  (SGHAT ocular model)
        rel = (np.abs(d["retinal_wcm2"] - d["corneal_wcm2"] * kf
                      / d["subtended_rad"] ** 2)
               / np.maximum(d["retinal_wcm2"], 1e-12)).max()
        check(f"{name}: E_r == E_c * (d_p/f)^2 / x^2", rel < 1e-9,
              f"max rel err {rel:.2e}")

        tau_all = d["atmos_transmittance"]
        check(f"{name}: 0 < tau <= 1 on every glare minute "
              "(beer_lambert active)",
              bool((tau_all > 0.0).all() and (tau_all <= 1.0).all()),
              f"min {tau_all.min():.4f} max {tau_all.max():.4f}")
        tau_means[(label, recp)] = float(tau_all.mean())

        # Attenuation scales the hazard exactly by tau per minute: reconstruct
        # the no-attenuation sibling from the same rows (E_c_no_atm =
        # DNI/1e4 * rho) and require the stored peak / its tau == the
        # no-attenuation peak on that same minute. obs_ptt @1.67 km keeps
        # ~72%, the ~16 km far field keeps < 5%.
        peak_tau = float(d.loc[d["retinal_wcm2"].idxmax(),
                               "atmos_transmittance"])
        row = d["retinal_wcm2"].idxmax()
        pred_no_tau = float(
            d.loc[row, "dni_wm2"] / 1e4 * d.loc[row, "reflectivity_used"]
            * kf / d.loc[row, "subtended_rad"] ** 2)
        no_tau = d["dni_wm2"] / 1e4 * d["reflectivity_used"] \
            * kf / d["subtended_rad"] ** 2
        check(f"{name}: peak / tau == reconstructed no-attenuation peak "
              "on the same minute",
              abs(peak / peak_tau - pred_no_tau) < 1e-6
              and abs(peak / peak_tau - float(no_tau.max())) < 1e-6,
              f"no-attenuation {pred_no_tau:.4f} (overall {no_tau.max():.4f})"
              f" -> attenuated {peak:.4f} W/cm2 (tau {peak_tau:.4f})")

    # Cross-case receiver ordering: receptor->array distance grows
    # obs_ptt (1.67 km) < west1_route_11 (4.84 km) < west1_route_2 (13.4 km)
    # < west1_route_1 (15.6 km), so the mean tau must fall monotonically even
    # though the four receptors span the two orientations. Tilt is
    # ground-distance-invariant, so the t10 labels stand in for all tilts.
    ordered = [("az098_e903_t10", "obs_ptt"),
               ("az098_e903_t10", "west1_route_11"),
               ("az278_e903_t10", "west1_route_2"),
               ("az278_e903_t10", "west1_route_1")]
    taus = [tau_means.get(k) for k in ordered]
    if all(t is not None for t in taus):
        line("  mean atmospheric transmittance by receptor distance:")
        for (lab, rec), t in zip(ordered, taus):
            line(f"    {lab}/{rec:<16} tau(mean) = {t:.4f}")
        check("cross-case: mean tau falls monotonically with receptor "
              "distance (1.67 > 4.84 > 13.4 > 15.6 km)",
              taus[0] > taus[1] > taus[2] > taus[3],
              " -> ".join(f"{t:.4f}" for t in taus))
    else:
        print("  [skip] tau ordering: case outputs missing")


def _detail(label, recp):
    path = os.path.join(REPO, "output", label, "results",
                        f"glare_hours_PV-A__{recp}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n# ---- validate_e903.py run "
                f"{pd.Timestamp.now()} ----\n")

    checks_model()
    curve_table()
    make_fig()
    case_selfcheck()

    res = (f"=== RESULT: {len(PASS)} passed, {len(FAIL)} failed "
           f"({pd.Timestamp.now()}) ===")
    line(res)
    print(f"\n{'=' * 56}\n{res}")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
    print("ALL E903-STYLE CHECKS PASSED")


if __name__ == "__main__":
    main()