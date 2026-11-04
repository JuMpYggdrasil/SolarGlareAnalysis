"""Validation of the IEC 62471 assessment layer (src/solar_glare/iec62471.py).

Units ranked as the primary hazard method alongside SGHAT. The checks cover:
  [A] Known-answer values: sun solid angle, source radiance from E_c,
      K_EYE-retinal round-trip, regime consistency.
  [B] Spectral weights kX: bounds, AM1.5 integral, stability of kR/kB.
  [C] Angular subtense (Sec 4.2.2): clamps ALPHA_MIN/MAX, arithmetic mean,
      retina-image bounds for the panel geometry.
  [D] FOV averaging (Sec 4.2.2/5.2-5.4): gamma(t) table values, dilution.
  [E] Table 6.1 emission limits + risk-group mapping (incl. the EN 62471
      small-source blue-light Exempt value 1.0 W/m2), decision-tree routing
      (alpha < 11 mrad -> Sec 4.3.4 E_B, else Sec 4.3.3 L_B).
  [F] Allowed-exposure duration inversions (Eqs 4.6/4.9 -> t_max).
  [G] End-to-end wiring: IEC columns on the per-pair detail rows, summary
      IEC aggregates, and the E_c <-> L / L_R <-> E_r consistency relations.
  [H] Smoke-tests of the IEC figures (risk zones, t_max, weighting functions).

Usage (from repo root):
    .venv\\Scripts\\python.exe scripts\\validate_iec62471.py
Results are appended to output/validation/iec62471.txt.
"""

import copy
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from solar_glare import iec62471 as iec
from solar_glare.geometry import sun_vector, surface_normal, reflect
from solar_glare.hazard import (
    SUN_SUBTENDED_RAD, PUPIL_DIAMETER, EYE_FOCAL_LENGTH, OCULAR_TRANSMISSION,
)
from solar_glare.analysis import load_config, run_analysis
from solar_glare import viz as _viz

OUT_PATH = os.path.join(REPO, "output", "validation", "iec62471.txt")
PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# [A] Known-answer optical values
# ---------------------------------------------------------------------------
print("\n[A] Known-answer optical values (sun disc / eye model)")
check("[A] Omega_sun = 2*pi*(1-cos(beta/2)) ~ 6.7929e-5 sr",
      close(iec.OMEGA_SUN, 6.7929e-5, 1e-8),
      f"{iec.OMEGA_SUN:.3e} sr")
check("[A] exact Omega_sun within 0.1% of small-angle pi*beta^2/4",
      abs(iec.OMEGA_SUN - np.pi * SUN_SUBTENDED_RAD ** 2 / 4.0) <
      1e-3 * np.pi * SUN_SUBTENDED_RAD ** 2 / 4.0)
K_EYE = (np.pi / 4.0) * (PUPIL_DIAMETER / EYE_FOCAL_LENGTH) ** 2 * OCULAR_TRANSMISSION
check("[A] K_EYE = (pi/4)(dp/f)^2 tau = 0.005434",
      close(iec.K_EYE, K_EYE, 1e-9) and close(iec.K_EYE, 0.005434, 1e-5),
      f"{iec.K_EYE:.6f}")
L_sun = iec.source_radiance(0.1)   # E_c = 0.1 W/cm2 == DNI 1000, rho = 1
check("[A] L = E_c*1e4/Omega_sun: DNI 1000 & rho=1 -> 1.472e7 W/m2/sr",
      close(L_sun, 1.4721e7, 1.0e3), f"{L_sun:.4e}")
check("[A] sun-disc retinal round-trip E_r = L*K_EYE/1e4 = 8.0 W/cm2",
      close(L_sun * iec.K_EYE / 1e4, 8.0, 0.08),
      f"{L_sun * iec.K_EYE / 1e4:.3f} W/cm2")

# ---------------------------------------------------------------------------
# [B] Spectral weighting factors (IEC 60904-3:2019 Table 1 direct, spectrally-flat rho)
# ---------------------------------------------------------------------------
print("\n[B] IEC 60904-3:2019 Table 1 direct spectral weights kX")
w = iec.spectral_weights()
wl, e = iec._spectrum()
tot = float(np.trapezoid(e, wl))
check("[B] IEC direct integral ~897.5 W/m2 (Table 1, 280-4000 nm)",
      890.0 < tot < 905.0, f"{tot:.1f} W/m2")
check("[B] IEC spectrum is 2002 points, 280.0-4000.0 nm",
      len(wl) == 2002 and wl[0] == 280.0 and wl[-1] == 4000.0,
      f"n={len(wl)} {wl[0]:.1f}-{wl[-1]:.1f}")
for k, v in w.items():
    upper = 1.5 if k == "kR" else 1.0  # kR > 1 is legitimate: Table 4.2 R(λ)
    check(f"[B] 0 < {k} < {upper}", 0.0 < v < upper, f"{v:.4f}")
check("[B] kR = R(lambda)-weighted IEC direct / full spectrum ~ 1.290 "
      "(Table 4.2 R has the blue-violet 10x hump at 435-440 nm)",
      close(w["kR"], 1.290, 0.02), f"{w['kR']:.4f}")
check("[B] kB = B(lambda)-weighted IEC direct / full spectrum ~ 0.0859",
      close(w["kB"], 0.0859, 0.008), f"{w['kB']:.4f}")
check("[B] skin fraction kH ~ 0.969 (most of the solar band is 380-3000 nm)",
      close(w["kH"], 0.969, 0.03), f"{w['kH']:.4f}")
check("[B] UVA fraction < IR fraction < skin fraction",
      w["kUVA"] < w["kIR"] < w["kH"],
      f"{w['kUVA']:.4f} < {w['kIR']:.4f} < {w['kH']:.4f}")
# weight() cache is stable across calls
check("[B] weights cached (id identical on 2nd call)",
      iec.spectral_weights() is w)

# [B2] The committed CSV must equal the reference extract (round-trip), so
# the standard document is the single source of truth for the spectrum.
print("\n[B2] IEC 60904-3:2019 Table 1 CSV vs reference extract")
from export_iec60904_3_spectrum import parse_table as _parse_iec_table
_txt = os.path.join(REPO, "references", "_BS_EN_IEC_60904-3-2019.txt")
_csv = os.path.join(REPO, "src", "solar_glare", "data",
                    "IEC60904-3_2019_Table1.csv")
_tab = _parse_iec_table(_txt)
_got = np.genfromtxt(_csv, delimiter=",", skip_header=1)
check("[B2] CSV row count == extract table (2002)",
      _got.shape == _tab.shape, f"{_got.shape} vs {_tab.shape}")
_abs = np.abs(_got - _tab)
check("[B2] CSV == extract to 1e-9 (all 7 columns)",
      np.max(_abs) <= 1e-9, f"max abs diff {np.max(_abs):.2e}")
check("[B2] CSV direct column == _spectrum()",
      np.array_equal(_got[:, 4], np.asarray(iec._spectrum()[1])) or
      np.max(np.abs(_got[:, 4] - np.asarray(iec._spectrum()[1]))) <= 1e-12,
      "direct columns agree")

# ---------------------------------------------------------------------------
# [C] Angular subtense Sec 4.2.2
# ---------------------------------------------------------------------------
print("\n[C] Angular subtense alpha (arithmetic mean of clamped dims)")
a1 = float(iec.angular_subtense(2.0, 4890.0, 2500.0))
check("[C] PV-A 2x4890 m @2.5 km -> alpha=0.05085 (height clamped 0.1, w clamped 0.0017)",
      close(a1, 0.05085, 1e-4), f"{a1:.5f} rad")
a2 = float(iec.angular_subtense(2.0, 4890.0, 10.0))
check("[C] @10 m -> alpha=0.1 (ALPHA_MAX cap)",
      close(a2, 0.1, 1e-6), f"{a2:.4f} rad")
a3 = float(iec.angular_subtense(2.0, 4890.0, 1.0e8))
check("[C] far field -> alpha=ALPHA_MIN=0.0017 (point-source floor)",
      close(a3, iec.ALPHA_MIN, 1e-9), f"{a3:.5f} rad")
check("[C] alpha never leaves [0.0017, 0.1]",
      np.all((np.array([a1, a2, a3]) >= iec.ALPHA_MIN - 1e-12) &
             (np.array([a1, a2, a3]) <= iec.ALPHA_MAX + 1e-12)))
a_geo = float(iec.angular_subtense(2.0, 4890.0, 973.0))
check("[C] @973 m: h/d>5 caps at 0.1, w/d=0.00206 (unclamped) -> 0.05103",
      close(a_geo, 0.05103, 1e-4), f"{a_geo:.5f} rad")

# ---------------------------------------------------------------------------
# [D] FOV averaging gamma(t) + dilution
# ---------------------------------------------------------------------------
print("\n[D] FOV averaging (Sec 4.2.2)")
check("[D] gamma(0.25 s) = 0.011 rad", close(float(iec.fov_gamma(0.25)), 0.011, 1e-9))
check("[D] gamma(100 s) = 0.011 rad", close(float(iec.fov_gamma(100.0)), 0.011, 1e-9))
g1000 = float(iec.fov_gamma(1000.0))
check("[D] gamma(1000 s) = 0.011*sqrt(10) = 0.034785",
      close(g1000, 0.011 * np.sqrt(10.0), 1e-9), f"{g1000:.6f}")
check("[D] gamma(1e4 s) = 0.1 rad", close(float(iec.fov_gamma(1.0e4)), 0.1, 1e-9))
check("[D] dilution(a=0.05, t=0.25s)=1 (fills 0.011-rad cone)",
      close(float(iec.fov_dilution(0.05, 0.25)), 1.0, 1e-9))
d_small = float(iec.fov_dilution(0.005, 0.25))
check("[D] dilution(a=0.005, 0.25s) = (0.005/0.011)^2 = 0.2066",
      close(d_small, (0.005 / 0.011) ** 2, 1e-9), f"{d_small:.4f}")
d_deep = float(iec.fov_dilution(0.002, 1.0e4))
check("[D] dilution(a=0.002, 1e4s) = (0.002/0.1)^2 = 4e-4",
      close(d_deep, 4.0e-4, 1e-9), f"{d_deep:.3e}")

# ---------------------------------------------------------------------------
# [E] Table 6.1 limits, decision tree, risk-group mapping
# ---------------------------------------------------------------------------
print("\n[E] Emission limits + risk-group mapping")
alpha = np.array([0.05])
ee = np.full(1, 1.0e6)  # arbitrary L_R
el_r1 = float((iec._THERMAL_EXEMPT_K / alpha)[0])
el_r2 = float((iec._THERMAL_RG2_K / alpha)[0])
check("[E] thermal EL tableau: Exempt/RG1=2.8e4/alpha=5.6e5, RG2=7.1e4/alpha=1.42e6",
      close(el_r1, 2.8e4 / 0.05, 1e-6) and close(el_r2, 7.1e4 / 0.05, 1e-6),
      f"{el_r1:.4e} / {el_r2:.4e} W/m2/sr")
# risk boundaries for blue-light radiance (Exempt 100, RG1 1e4, RG2 4e6)
rb = np.array([100.0, 500.0, 1e4 + 0.1, 4e6 + 0.1])
lab = iec.risk_group(rb, (iec.BLUE_EXEMPT_RAD, iec.BLUE_RG1_RAD, iec.BLUE_RG2_RAD))
check("[E] blue radiance groups: [Exempt, RG1, RG2, RG3]",
      list(lab) == ["Exempt", "RG1", "RG2", "RG3"], str(list(lab)))
# small-source blue-light irradiance (EN edition Exempt = RG1 = 1.0 W/m2, so the
# RG1 band is empty: any value above 1.0 up to 400 jumps straight to RG2)
eb = np.array([1.0, 5.0, 300.0, 410.0])
lab = iec.risk_group(eb, (iec.BLUE_EXEMPT_E, iec.BLUE_RG1_E, iec.BLUE_RG2_E))
check("[E] blue small-source E_B: [Exempt, RG2, RG2, RG3] (EN Exempt==RG1=1.0)",
      list(lab) == ["Exempt", "RG2", "RG2", "RG3"], str(list(lab)))
# decision tree routing: alpha<11 mrad -> Sec 4.3.4 E_B; >=11 -> 4.3.3 L_B
ev = iec.evaluate_glare(np.array([0.1]), np.array([8.0]), np.array([2000.0]),
                        0.5, 0.5, abs_cos_ref=1.0)
small = ev["alpha_rad"][0] < iec.ALPHA_SMALL
exp_blue = iec.risk_group(np.array([ev["eb_w_m2"][0]]),
                          (iec.BLUE_EXEMPT_E, iec.BLUE_RG1_E, iec.BLUE_RG2_E))[0]
check("[E] decision tree: small source (alpha<11 mrad) judged by E_B, "
      f"risk_blue={ev['risk_blue'][0]}, expected {exp_blue}",
      small and ev["risk_blue"][0] == exp_blue,
      f"alpha={ev['alpha_rad'][0]:.4f} E_B={ev['eb_w_m2'][0]:.1f} W/m2")
# thermal RG boundary sweep (alpha = 0.05): Exempt == RG1 = 2.8e4/alpha = 5.6e5,
# RG2 = 7.1e4/alpha = 1.42e6 -> the RG1 band is empty (Exempt==RG1 like blue small)
th = np.array([5.6e5, 5.6e5 + 1.0, 1.42e6 + 1.0])
lab = iec.risk_group(th, (5.6e5, 5.6e5, 1.42e6))
check("[E] thermal radiance groups: [Exempt, RG2, RG3] (Exempt==RG1)",
      list(lab) == ["Exempt", "RG2", "RG3"], str(list(lab)))
# formal risk_eye = max(thermal, blue)
m = iec.evaluate_glare(np.array([0.1]), np.array([8.0]), np.array([50.0]),
                       2.0, 4890.0)
check("[E] formal eye group is the max of thermal+blue (RG3 here)",
      m["risk_eye"][0] == "RG3" and m["risk_thermal"][0] == "RG3",
      f"thermal={m['risk_thermal'][0]} blue={m['risk_blue'][0]} eye={m['risk_eye'][0]}")

# ---------------------------------------------------------------------------
# t_max inversions (Eq 4.6 / 4.9)
# ---------------------------------------------------------------------------
print("\n[F] Allowed-exposure duration inversions")
t1 = float(iec._tmax_thermal(np.array([1.0e6]), np.array([0.05]))[0])
check("[F] tmax_thermal: L_R*alpha=5e4 -> t=1.0 s exactly",
      close(t1, 1.0, 1e-6), f"{t1:.4f} s")
t2 = float(iec._tmax_thermal(np.array([2.0e6]), np.array([0.05]))[0])
check("[F] tmax_thermal: L_R*alpha=1e5 -> t=0.0625 s (Eq 4.9 inversion)",
      close(t2, 0.0625, 1e-6), f"{t2:.4f} s")
t3 = float(iec._tmax_blue(np.array([1.0e6]))[0])
check("[F] tmax_blue: L_B=1e6 -> t=1 s (Eq 4.6 inversion)",
      close(t3, 1.0, 1e-9), f"{t3:.3e} s")
t4 = float(iec._tmax_blue(np.array([1.0e4]))[0])
check("[F] tmax_blue: L_B=1e4 -> t=100 s",
      close(t4, 100.0, 1e-6), f"{t4:.3e} s")

# ---------------------------------------------------------------------------
# [G] End-to-end wiring (real config, hourly, IEC enabled)
# ---------------------------------------------------------------------------
print("\n[G] End-to-end wiring: IEC columns on detail + summary aggregates")
cfg = load_config(os.path.join(REPO, "config", "site.yaml"))
cfg["environment"]["iec62471_enabled"] = True
cfg["site"]["time_freq"] = "h"          # fast wire test
cfg["arrays"] = [cfg["arrays"][0]]      # single array
cfg["arrays"][0]["azimuth_deg"] = 98.0  # the committed az098 case -> obs_ptt hits
t_start = time.time()
times, _, _, long_df, summary, details, site, arrays, receptors = run_analysis(copy.deepcopy(cfg))
elapsed = time.time() - t_start
_iec_cols = [c for c in summary.columns if c.startswith("iec_")]
check("[G] summary exposes iec_* aggregate columns",
      len(_iec_cols) >= 4, f"{_iec_cols}")
hit_pairs = [(a, r) for (a, r), d in details.items() if not d.empty]
check("[G] analysis ran ({:.1f}s) and at least one pair reports glare rows"
      .format(elapsed), len(hit_pairs) >= 1,
      f"{len(hit_pairs)} of {len(details)} pairs")
if hit_pairs:
    a, r = hit_pairs[0]
    d = details[(a, r)]
    row = d.iloc[0]
    print(f"      sample pair {a}|{r}: {len(d)} rows, cols={[c for c in d.columns if c.startswith('iec_')][:10]}...")
    check("[G] IEC columns present on detail rows",
          all(c in d.columns for c in (
              "iec_alpha_rad", "iec_l_w_sr_m2", "iec_lr_w_sr_m2",
              "iec_lb_w_sr_m2", "iec_eb_w_m2", "iec_tmax_thermal_s",
              "iec_tmax_blue_s", "iec_risk_thermal", "iec_risk_blue",
              "iec_risk_eye", "iec_er_lr_wcm2")),
          str([c for c in d.columns if c.startswith('iec_')])[:160])
    check("[G] risk columns hold only Exempt/RG1/RG2/RG3",
          d["iec_risk_eye"].isin(["Exempt", "RG1", "RG2", "RG3"]).all())
    ok_rho = np.allclose(
        d["iec_l_w_sr_m2"], d["corneal_wcm2"] * 1e4 / iec.OMEGA_SUN, rtol=1e-9)
    check("[G] L == E_c*1e4/Omega (exact identity to the pipeline)",
          ok_rho)
    thr = d["iec_el_thermal_rg2_w_sr_m2"]
    check("[G] tmax_thermal = (5e4/(L_R*alpha))^4 round-trips the RG2 EL",
          np.allclose(d["iec_tmax_thermal_s"],
                      np.power(iec.THERMAL_EL_K / (d["iec_lr_w_sr_m2"] * d["iec_alpha_rad"]), 4.0),
                      rtol=1e-9))
    kr = w["kR"]
    rel = np.abs(d["iec_er_lr_wcm2"] - kr * d["iec_l_w_sr_m2"] * iec.K_EYE / 1e4)
    check("[G] bijection kR*E_r == kR*L*K_EYE/1e4 (er_lr consistent, <2% for sun-disc)",
          float(rel.max()) < 0.02 * float(kr * d["retinal_wcm2"].max()),
          f"max rel dev {float(rel.max()):.3f}")
    check("[G] every glare minute has a rated eye risk (no NaN)",
          d["iec_risk_eye"].notna().all() and d["iec_tmax_thermal_s"].notna().all())
    # sanity: reflected beam cannot be brighter than the sun (L <= rho*tau*L_sun)
    expected = d["reflectivity_used"]
    if "atmos_transmittance" in d.columns:
        expected = expected * d["atmos_transmittance"]
    sun_lim = expected * iec.source_radiance(0.1)  # rho*tau * L at DNI 1000
    check("[G] L <= rho*L_sun (beam brightness bounded by the sun)",
          bool(np.all(d["iec_l_w_sr_m2"] <= sun_lim * 1.05)),
          f"max L {float(d['iec_l_w_sr_m2'].max()):.3e} vs {float(sun_lim.max()):.3e}")
    if not summary.empty:
        srow = summary.loc[summary["iec_worst_risk"].notna()].head(1)
        check("[G] summary fills iec_worst_risk / iec_min_tmax rows for hit pairs",
              len(srow) == 1 and not pd.isna(srow["iec_min_tmax_thermal_s"].iloc[0]),
              f"rows={len(srow)}")

    # ------------------------------------------------------------------
    # [H] Smoke-test the IEC figure renderers on the same hourly data
    # ------------------------------------------------------------------
    iec_out_dir = os.path.dirname(OUT_PATH)
    zones_path = os.path.join(iec_out_dir, "iec62471_zones.png")
    _viz.plot_iec62471_zones(details, zones_path)
    _sz1 = os.path.getsize(zones_path) if os.path.exists(zones_path) else 0
    check("[H] IEC 62471 zones figure rendered",
          _sz1 > 5000, f"{_sz1 / 1024:.1f} KB")
    _viz.plot_iec62471_tmax(details,
                            os.path.join(iec_out_dir, "iec62471_tmax_thermal.png"),
                            os.path.join(iec_out_dir, "iec62471_tmax_blue.png"))
    _sz2a = os.path.getsize(os.path.join(iec_out_dir, "iec62471_tmax_thermal.png"))
    _sz2b = os.path.getsize(os.path.join(iec_out_dir, "iec62471_tmax_blue.png"))
    check("[H] IEC 62471 tmax figures rendered (thermal + blue)",
          _sz2a > 5000 and _sz2b > 5000,
          f"thermal {_sz2a / 1024:.1f} KB / blue {_sz2b / 1024:.1f} KB")
    _wts_path = os.path.join(iec_out_dir, "iec62471_weights.png")
    _viz.plot_iec62471_weights(_wts_path)
    _sz3 = os.path.getsize(_wts_path) if os.path.exists(_wts_path) else 0
    check("[H] IEC 62471 weighting functions figure rendered",
          _sz3 > 3000, f"{_sz3 / 1024:.1f} KB")
    _sp_path = os.path.join(iec_out_dir, "iec62471_spectrum_weighted.png")
    _viz.plot_iec62471_spectrum_weighted(_sp_path)
    _sz4 = os.path.getsize(_sp_path) if os.path.exists(_sp_path) else 0
    check("[H] IEC 62471 spectrum-before/after-weighting figure rendered",
          _sz4 > 3000, f"{_sz4 / 1024:.1f} KB")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
stamp = time.strftime("%Y-%m-%d %H:%M:%S")
with open(OUT_PATH, "a", encoding="utf-8") as f:
    f.write(f"\n--- IEC 62471 validation {stamp} ---\n")
    for name in PASS:
        f.write(f"  [OK] {name}\n")
    for name in FAIL:
        f.write(f"  [FAIL] {name}\n")
    f.write(f"  RESULT: {len(PASS)} passed, {len(FAIL)} failed, "
            f"elapsed {elapsed:.1f}s (wire run)\n")

print(f"\n{'=' * 56}")
print(f"VALIDATION COMPLETE: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL IEC 62471 CHECKS PASSED")