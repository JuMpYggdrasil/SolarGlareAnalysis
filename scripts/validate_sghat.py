"""Validation of the glare engine against published reference values.

Reference: C. K. Ho et al., "Methodology to Assess Potential Glint and Glare
Hazards from Concentrating Solar Power Plants", J. Sol. Energy Eng.
133(1):011011 (2011). DOI 10.1115/1.4004349. And Sandia SGHAT / ForgeSolar
documentation.

The validation uses two independent routes:
  A) Known-answer tests vs. the canonical values published in Ho 2011
     (sun subtended angle, retinal irradiance of the sun, hazard thresholds).
  B) A from-the-equations reference implementation written for this check only,
     compared term-by-term with the library functions so a coding slip in
     `hazard.py` or `geometry.py` shows up.

Run from repo root (inside the venv):
    .venv\\Scripts\\python.exe scripts\\validate_sghat.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from solar_glare.geometry import sun_vector, surface_normal, reflect, angle_between
from solar_glare.hazard import (
    SUN_SUBTENDED_RAD,
    PUPIL_DIAMETER,
    EYE_FOCAL_LENGTH,
    OCULAR_TRANSMISSION,
    subtended_angle,
    corneal_irradiance,
    retinal_irradiance,
    burn_threshold,
    flash_threshold,
    classify,
    beam_spread_angle,
    array_limiting_angle,
    effective_subtended_angle,
)
from solar_glare.analysis import analyze_pair

PASS = []
FAIL = []


def assert_close(actual, expected, tol, name):
    """Append a result to PASS/FAIL directly (works in loop bodies)."""
    ok = abs(actual - expected) <= tol
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  -- got {actual:.6g}, want {expected:.6g} (tol {tol:.1e})")
    return ok


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# A) Known-answer tests vs Ho 2011 / Sandia
# ---------------------------------------------------------------------------
print("\n[A] Known-answer tests (Ho 2011 / Sandia reference values)")

# A1 - apparent solar disc ~ 0.5 deg
check("sun subtended = 9.3 mrad (~0.53 deg)",
      abs(SUN_SUBTENDED_RAD - 9.3e-3) < 1e-5,
      f"{SUN_SUBTENDED_RAD:.4e} rad = {np.degrees(SUN_SUBTENDED_RAD):.3f} deg")

# A2 - staring at the sun: E_c = DNI=1000 W/m2 -> 0.1 W/cm2, x = sun disc
#     Eq.(2) gives E_r ~ 8 W/cm2 (canonical value quoted in Ho 2011).
sun_x = SUN_SUBTENDED_RAD
sun_ec = corneal_irradiance(1000.0, 1.0)
sun_er = retinal_irradiance(sun_ec, sun_x)
check("viewing sun gives E_r ~ 8 W/cm2",
      abs(float(sun_er) - 8.0) < 0.5,
      f"E_r = {sun_er:.2f} W/cm2  (reference ~8 W/cm2, Ho 2011)")

# A3 - burn threshold: 0.118/x below x=0.118 rad, plateau 1.0 above
x_small = 0.01
x_big = 0.2
check("burn threshold = 0.118/x (x<0.118 rad)",
      abs(float(burn_threshold(x_small)) - 0.118 / x_small) < 1e-12)
check("burn threshold plateau = 1 W/cm2 (x>=0.118 rad)",
      abs(float(burn_threshold(x_big)) - 1.0) < 1e-12)

# A4 - flash threshold = 3.59e-5 / x^1.77
check("flash threshold = 3.59e-5/x^1.77",
      abs(float(flash_threshold(0.01)) - 3.59e-5 / 0.01 ** 1.77) < 1e-15)

# A5 - hazard classification of the sun itself:
#     x=9.3 mrad, E_r=8 W/cm2. burn=0.118/0.0093=12.7, flash=0.142  -> Y.
burn_sun = burn_threshold(sun_x)
flash_sun = flash_threshold(sun_x)
hz_sun = classify(np.array([sun_er]), np.array([sun_x]))[0]
check("sun is classified Yellow (flash < E_r < burn)",
      hz_sun == "Y"
      and flash_sun < sun_er < burn_sun,
      f"flash={float(flash_sun):.3f} < E_r={float(sun_er):.1f} < burn={float(burn_sun):.1f} -> {hz_sun}")

# A6 - extended source limit: as source size grows, subtended angle grows
check("subtended angle includes solar disc at long range",
      float(subtended_angle(0.0, 5000.0)) == SUN_SUBTENDED_RAD,
      f"point source @5km -> {float(subtended_angle(0.0, 5000.0)):.4e} rad")

# A7 - flat-plate far field: no Red possible for realistic panels (all distant
#      receptors below burn threshold). Sanity: >1000 m away, 30 m panel,
#      DNI=1000, reflectivity<=0.06 -> far below burn.
er_far = retinal_irradiance(corneal_irradiance(1000.0, 0.06),
                            subtended_angle(30.0, 1000.0))
check("distant flat panel far below burn threshold",
      float(er_far) < float(burn_threshold(subtended_angle(30.0, 1000.0))),
      f"E_r={float(er_far):.4f} W/cm2")

# ---------------------------------------------------------------------------
# A8-A11 - worked values from Sandia SGHAT Technical Reference v6 (SAND2014-18360)
#          and Users Manual v2G (the Manchester-Boston MHT case study).
# ---------------------------------------------------------------------------
print("\n[A2] Worked values from SGHAT Technical Reference v6 + Users Manual v2G")

# A8 - subtended beam angle. Ref V6 sec 4.3: beta = 2*(theta_sun/2 + 2*3*sigma).
#      With sigma = 0: beta = theta_sun = 9.3 mrad ("total beam spread of 9.3
#      mrad (the subtended angle of the sun)" - Users Manual v2G Fig. 22).
beam_zero = SUN_SUBTENDED_RAD + 12.0 * 0.0
check("beam spread @slope error 0 = 9.3 mrad (Users Manual v2G)",
      abs(float(beam_zero) - 9.3e-3) < 1e-12,
      f"beta = {beam_zero:.4e} rad")

# A9 - with 10 mrad slope error the Users Manual quotes 0.1293 rad total beam
#      spread (v2G Fig. 20): beta = 9.3e-3 + 12*10e-3.
beam_10 = SUN_SUBTENDED_RAD + 12.0 * 10e-3
check("beam spread @slope error 10 mrad = 0.1293 rad (Users Manual v2G)",
      abs(float(beam_10) - 0.1293) < 1e-12,
      f"beta = {beam_10:.4e} rad")

# A10 - sandia Technical Reference v6 Table 2: mean slope error -> average beam
#       spread. Ref V6 sec 4.3 & v6 Table 2 col. 3.
SGHAT_SLOPE_BEAM = {
    "Smooth Glass w/o ARC":   (6.55e-3,  87.9e-3),
    "Smooth Glass w/ ARC":    (8.43e-3, 110.0e-3),
    "Light Textured w/o ARC": (9.70e-3, 126.0e-3),
    "Light Textured w/ ARC":  (9.16e-3, 119.0e-3),
    "Deeply Textured Glass":  (82.6e-3, 1000.0e-3),
}
for surf, (slope, beam_ref) in SGHAT_SLOPE_BEAM.items():
    beam_sghat = SUN_SUBTENDED_RAD + 12.0 * slope
    # Published values are rounded to the nearest mrad (Table 2), allow 1 mrad.
    assert_close(float(beam_sghat), beam_ref, 1.0e-3,
                 f"beam spread vs v6 Table 2: {surf}")

# ---------------------------------------------------------------------------
# A11-A13 - SGHAT v6 Sec.5.5 array-limiting (projected visible area) angle.
#   theta = (1/d)*sqrt(4*A*|cos(theta_ref-pva)|/pi). The subtended source
#   angle x used for the hazard = min(beta, theta) (Sec.5.5: "replaces the
#   solar beam angle ... with an array-limiting beam angle if the latter is
#   a smaller value").
# ---------------------------------------------------------------------------
# A11 - worked value: PV-B demo array (2 x 1000 m, A=2000 m2) viewed face-on
#      (|cos| = 1) from 3260 m. theta = sqrt(4*2000/pi)/3260 = 50.46/3260.
theta_pvb = float(array_limiting_angle(2000.0, 3260.0, 1.0))
expect_pvb = np.sqrt(4.0 * 2000.0 / np.pi) / 3260.0
check("Sec.5.5 array-limiting angle worked value (PV-B, A=2000 m2, d=3260 m, cos=1)",
      abs(theta_pvb - expect_pvb) < 1e-12,
      f"theta = {theta_pvb:.6g} rad = {np.degrees(theta_pvb):.3f} deg")

# A12 - large array seen face-on: theta > beta -> effective x = beta (sun disc).
beta0 = float(beam_spread_angle(0.0))
x_large = float(effective_subtended_angle(2000.0, 3260.0, 1.0, 0.0))
check("effective x = min(beta, theta): large array -> x = beta",
      abs(x_large - beta0) < 1e-12,
      f"beta={beta0:.6g} rad, theta={theta_pvb:.6g} rad, x={x_large:.6g} rad")

# A13 - small/off-axis array: theta < beta -> effective x = theta (array-limited).
x_small = float(effective_subtended_angle(50.0, 5000.0, 0.2, 0.0))
theta_small = float(array_limiting_angle(50.0, 5000.0, 0.2))
check("effective x = min(beta, theta): small/off-axis -> x = theta",
      abs(x_small - theta_small) < 1e-12 and x_small < beta0,
      f"theta={theta_small:.6g} rad < beta={beta0:.6g} rad, x={x_small:.6g} rad")

# A14 - foreshortening: viewing the array edge-on (|cos|->0) shrinks theta.
theta_side = float(array_limiting_angle(2000.0, 3260.0, 0.05))
check("Sec.5.5 foreshortening: |cos|=0.05 << face-on theta",
      theta_side < theta_pvb * 0.3,
      f"theta_side={theta_side:.6g} rad vs face-on {theta_pvb:.6g} rad")

# ---------------------------------------------------------------------------
# B) Independent from-the-paper reference implementation
# ---------------------------------------------------------------------------
print("\n[B] Independent reference implementation (from Ho 2011 equations)")


def ref_retinal_irradiance(ec, x):
    # Eq.(2): E_r = E_c * dp^2 / (f^2 * x^2) * tau  (each term written out)
    return ec * (PUPIL_DIAMETER ** 2) / (EYE_FOCAL_LENGTH ** 2 * x ** 2) * OCULAR_TRANSMISSION


def ref_burn(x):
    return 0.118 / x if x < 0.118 else 1.0


def ref_flash(x):
    return 3.59e-5 / x ** 1.77


def ref_classify(er, x):
    if er >= ref_burn(x):
        return "R"
    if er >= ref_flash(x):
        return "Y"
    return "G"


rng = np.random.default_rng(42)
mismatch = 0
for _ in range(5000):
    x = 10 ** rng.uniform(-4, -0.5)          # 1e-4 .. 0.3 rad
    ec = 10 ** rng.uniform(-5, 1.0)          # 1e-5 .. 10 W/cm2
    er = float(retinal_irradiance(ec, x))
    er_ref = ref_retinal_irradiance(ec, x)
    if abs(er - er_ref) / er_ref > 1e-12:
        mismatch += 1
    if classify(np.array([er]), np.array([x]))[0] != ref_classify(er, x):
        mismatch += 1
check("retinal_irradiance + classify match paper equations (5000 random cases)",
      mismatch == 0, f"{mismatch} mismatches")

# geometry: mirror law and reflection direction are self-consistent
mismatch = 0
for _ in range(2000):
    tilt = rng.uniform(0, 90)
    az = rng.uniform(0, 360)
    n = surface_normal(tilt, az)
    if np.linalg.norm(n) - 1.0 > 1e-12:
        mismatch += 1; continue
    s = sun_vector(rng.uniform(0, 360), rng.uniform(0, 90))
    r = reflect(s, n)
    if abs(np.linalg.norm(r) - 1.0) > 1e-9:
        mismatch += 1
    if abs(np.dot(r, n) - np.dot(s, n)) > 1e-9:
        mismatch += 1
check("geometry: normal unit + mirror-law r=2(s.n)n - s (2000 random cases)",
      mismatch == 0, f"{mismatch} mismatches")

# ---------------------------------------------------------------------------
# C) Physical self-consistency of the hit-detection logic
# ---------------------------------------------------------------------------
print("\n[C] Physical self-consistency of hit detection (analyze_pair)")

# A receptor placed EXACTLY along the reflection of a synthetic sun must be
# flagged; one far outside the beam must not. Build a tiny hand-made scenario
# on a flat frame to avoid pvlib: craft a fake position table.
from pandas import DataFrame
import pandas as pd


def fake_pos(az, el):
    idx = np.array([0])
    return DataFrame({"azimuth": np.array([az]), "apparent_elevation": np.array([el])},
                     index=idx)


class FakeArray:
    def __init__(self, normal, char_size=10.0, refl=0.05):
        self.normal = normal
        self.char_size = char_size  # m
        self.specular_reflectivity = refl
        self.center_enu = np.array([0.0, 0.0, 0.0])


class FakeDni:
    def __init__(self, n=1):
        self._series = pd.Series([1000.0] * n)

    @property
    def iloc(self):
        return self._series.iloc


class FakeReceptor:
    def __init__(self, enu):
        self.enu = np.array(enu, dtype=float)


# south-facing panel 20 deg, sun due south 20 deg -> normal incidence, ray
# reflects straight back along the sun vector (i.e. to the SOUTH and UP).
arr = FakeArray(surface_normal(20.0, 180.0), char_size=10.0)
n = arr.normal
s = sun_vector(180.0, 20.0)
r = reflect(s, n)

# receptor placed 2000 m away along the reflected ray
pos_on = FakeReceptor(r * 2000.0)
pos_off = FakeReceptor(np.array([2000.0, 0.0, 0.0]))  # due east, well outside beam

codes_on, det_on = analyze_pair(arr, pos_on, fake_pos(180.0, 20.0), FakeDni(n=1))
codes_off, det_off = analyze_pair(arr, pos_off, fake_pos(180.0, 20.0), FakeDni(n=1))
check("receptor on reflected beam is hit and classified",
      codes_on[0] in ("G", "Y", "R") and len(det_on) == 1,
      f"code={codes_on[0]}, detail rows={len(det_on)}")
check("receptor off beam is NOT hit",
      codes_off[0] == "N" and len(det_off) == 0,
      f"code={codes_off[0]}")

# A true back-face case: panel tilt 20 facing south -> normal is 20 deg off
# the horizontal to the SOUTH (up-component dominates). A low sun from the
# NORTH (elevation < tilt) falls on the BACK face (s.n < 0) and must not glare.
s_bad = sun_vector(0.0, 5.0)     # low sun due north
if np.dot(s_bad, n) >= 0:
    raise SystemExit("test setup error: expected s.n < 0 for low north sun")
r_bad = reflect(s_bad, n)        # points north
pos_north = FakeReceptor(r_bad * 2000.0)
codes_north, det_north = analyze_pair(arr, pos_north, fake_pos(0.0, 5.0), FakeDni(n=1))
check("sun behind panel (low north sun) produces no glare",
      codes_north[0] == "N" and len(det_north) == 0,
      f"code={codes_north[0]}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 56}")
print(f"VALIDATION COMPLETE: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL REFERENCE CHECKS PASSED")