"""Cross-check: references/_BS_EN_62471_2008.txt (hand-edited, normalized) vs the constants
and formulas implemented in src/solar_glare/iec62471.py.

This is the automated, persistent form of the hand cross-check done in the
pad/thai review: every constant, threshold and equation the CODE relies on is
asserted to appear correctly in the normalized text of the actual standard
(BS EN 62471:2008). It also re-derives the code's own numbers from the
ANALYTIC formulas the text states, so a drift in either direction is caught:

    * [A] angular definitions (Sec 4.2.2): alpha_min 0.0017 rad, alpha_max
          0.1 rad, small-source 0.011 rad, clamp-before-arithmetic-mean,
          alpha_eff(t) ramps.
    * [B] Table 4.2 B(lambda) / R(lambda): all 40 wavelength rows (300-495 nm)
          parsed from the text and compared with the code tables _B_WEIGHT /
          _R_WEIGHT knot-for-knot, plus the analytic tail bands
          (500-600/600-700 for B, 700-1050/1050-1150/1150-1200/1200-1400 for
          R) matched against the code branch formulas.
    * [C] Table 4.1 S_UV(lambda) -- REFERENCE-ONLY (see INFO note C7: the code
          deliberately uses the Figure 4.1 box approximation, not these
          discrete values). Guards against transcription drift on re-extract.
    * [D] Table 5.4 irradiance ELs (30/t, 10000/t, 100/t, 18000/t^0.75,
          20000/t^0.75, aperture 1.4 (80)).
    * [E] Equations 4.1-4.11 verbatim (constants 30/10000/10/1e6/100/50000/
          18000/20000, alpha t^0.25 exponent, t_max inversions).
    * [F] Table 6.1 all 8 hazard-group rows (UV, UVA, L_B, E_B small, L_R
          thermal, L_IR weak, E_IR + units).
    * [G] Table 5.5 radiance ELs (10^6/t x3, 100, 50000/(alpha*t^0.25),
          6000/alpha) + FOV formulas 0.011*sqrt(t/10), 0.0011*sqrt(t), 0.1.
    * [H] code self-consistency: 2.8e4/7.1e4 alpha constants re-derived from
          Eq 4.9 at t=10 s / t=0.25 s; fov_gamma()/blue_light_el()/
          blue_light_small_el()/retinal_thermal_el() sampled against the
          text's stated formulas.
    * [J] PD IEC/TR 62471-2:2009 provenance cross-check: the stored TR extract
      is manufacturer guidance (hazard distance / acceptance angles / labelling)
      built on the same IEC 62471:2006 limit table, does NOT carry any
      0.01 W/m^2 small-source alternative (the bare "0.01" tokens are log-axis
      ticks only), and therefore does not perturb the code's Table 6.1 values.

INFO notes (not pass/fail) document the deliberate simplifications and their
(provably benign) scope:
    * FOV gamma collapsed to 0.011 rad for t <= 100 s (the text only ramps
      alpha_min*sqrt(t/0.25) between 0.25 s and 10 s; irrelevant wherever
      alpha >= gamma because dilution saturates at 1).
    * Table 4.1 discrete S_UV vs the code's Figure 4.1 box approximation
      (kS is supplementary-only and the actinic clause is out-of-scope for
      the sun anyway).
    * Weak-visual-stimulus L_IR 6000/alpha (non-GLS) not implemented as a
      separate clause: the code uses the R(λ) thermal radiance for the
      FORMAL group only.
    * k_R/k_B are COMPUTED (AM1.5 x Table 4.2), not transcribed from the
      standard; values printed for the record.

Usage:
  .venv\\Scripts\\python.exe scripts\\validate_norm_vs_code.py
Results appended to output/validation/norm_vs_code.txt.
"""

import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from solar_glare import iec62471 as IEC

TXT = os.path.join(REPO, "references", "_BS_EN_62471_2008.txt")
OUT_PATH = os.path.join(REPO, "output", "validation", "norm_vs_code.txt")

PASS = 0
FAIL = 0
INF = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        tag = "OK "
    else:
        FAIL += 1
        tag = "FAIL"
    line = "  [%s] %s" % (tag, name) + ("  -- %s" % (detail,) if detail else "")
    print(line)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def info(name, detail=""):
    global INF
    INF += 1
    line = "  [INF] %s" % name + ("  -- %s" % detail if detail else "")
    print(line)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def section(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70 + "\n")


def slice_between(marker_a, marker_b):
    """Return (region_text, ok) for text BETWEEN the two literal markers."""
    i = text.find(marker_a)
    if i < 0:
        return "", False
    i += len(marker_a)  # start AFTER the marker so its own numbers are excluded
    j = text.find(marker_b, i)
    if j < 0:
        return "", False
    return text[i:j], True


def numbers(region):
    """Pure decimal tokens in a region ('313*' allowed; strips the star)."""
    return [t.rstrip("*") for t in region.split()
            if re.fullmatch(r"\d+(?:\.\d+)?\*?", t)]


def needle(name, needle_str):
    """Exact (whitespace-collapsed) substring must occur in the text."""
    s = re.sub(r"\s+", "", needle_str)
    ok = s in flat
    check(name, ok, ("needle '%s' found" % s) if ok
          else ("needle '%s' NOT FOUND" % s))


text = open(TXT, encoding="utf-8").read()
flat = re.sub(r"\s+", "", text)  # collapse all whitespace for literal checks


# --------------------------------------------------------------------------
# [A] Angular definitions (Sec 4.2.2) ---------------------------------------
# --------------------------------------------------------------------------
def check_angular():
    section("[A] Angular definitions (Sec 4.2.2)")
    needle("A1  alpha_min = 0.0017 radian", "0.0017radian")
    needle("A2  alpha_max = 0.1 radian (all retinal hazards)", "α_max,is0.1radianforallretinalhazards")
    needle("A3  small-source threshold alpha < 0.011 radian", "α<0.011radian")
    needle("A4  clamp dims to alpha_min/alpha_max BEFORE arithmetic mean",
           "prior to the determination of the arithmetic mean")
    needle("A5  arithmetic mean of max+min dimensions",
           "arithmetic mean of the maximum and minimum angular dimensions")
    needle("A6  worked example Z = (20+3)/2 = 11.5 mm", "Z = (20+3)/2 = 11.5 mm")
    needle("A7  alpha_eff ramp 0.25-10 s: alpha_min*sqrt(t/0.25)", "α_min*sqrt(t/0.25)")
    needle("A8  alpha_eff = 0.011 rad for t < 100 s (blue)", "0.011radianfortimeslessthan100s")
    needle("A9  alpha_eff = 0.011*sqrt(t/100) for 100 s < t < 10000 s",
           "0.011*sqrt(t/100)")
    needle("A10 alpha_eff = 0.1 rad for t > 10000 s (alpha_max)", "α_effissetto0.1radian")
    needle("A11 thermal EL FOV cone 0.0017..0.1 radian (Eq 4.9 note 1)",
           "notlessthan0.0017radian,andnotmorethan0.1radian,includedangle")
    # decision tree (Sec 4.2.2 / clause 4.3.4) mirrors code ALPHA_SMALL
    needle("A12 'less than 0.011 radian' -> small-source branch (clause 4.3.4)",
           "light source subtending an angle less than 0.011 radian")
    check("A13 code ALPHA_SMALL == 0.011 rad (text small-source threshold)",
          abs(IEC.ALPHA_SMALL - 0.011) < 1e-12, repr(IEC.ALPHA_SMALL))
    check("A14 code ALPHA_MIN == 0.0017 / ALPHA_MAX == 0.1 (text values)",
          abs(IEC.ALPHA_MIN - 0.0017) < 1e-12 and abs(IEC.ALPHA_MAX - 0.1) < 1e-12,
          "%r/%r" % (IEC.ALPHA_MIN, IEC.ALPHA_MAX))


# --------------------------------------------------------------------------
# [B] Table 4.2 B(lambda)/R(lambda) vs code tables --------------------------
# --------------------------------------------------------------------------
def check_table42():
    section("[B] Table 4.2 B(lambda) / R(lambda) vs iec62471 tables")
    sec, ok_b = slice_between("Table 4.2 Spectral weighting functions",
                              "For a weighted source radiance")
    check("B1  Table 4.2 region delimited", ok_b)
    toks = numbers(sec)
    rows = {}
    i = 0
    for want in range(300, 501, 5):
        if i >= len(toks) or not toks[i].isdigit():
            break  # band tail ("500-600" et al.) reached: no more knots
        lam = int(toks[i])
        if lam != want:
            break
        i += 1
        bv = float(toks[i])
        i += 1
        rv = None
        if lam >= 380 and i < len(toks):
            rv = float(toks[i])
            i += 1
        rows[lam] = (bv, rv)
    # This edition prints NO standalone 500 knot: the last row is 495 and the
    # "500-600" band row starts with B = 10^((450-lambda)/50) (== 0.1 at 500).
    check("B2  all 40 wavelength rows parsed (300-495 nm, no 500 knot)",
          len(rows) == 40,
          "%d rows: %s..%s" % (len(rows), min(rows) if rows else "-",
                               max(rows) if rows else "-"))
    if len(rows) != 40:
        return
    bad = 0
    for lam, (b, r) in rows.items():
        cb = float(IEC._b_weight(float(lam)))
        if abs(cb - b) > 1e-6:
            bad += 1
        if lam >= 380:
            cr = float(IEC._r_weight(float(lam)))
            if abs(cr - r) > 1e-6:
                bad += 1
    check("B3  B(lambda) + R(lambda) text rows == code lookup (knot-by-knot)",
          bad == 0, "%d mismatches" % bad)
    # code B/R tables are exactly the text knots (no drift inside iec62471.py)
    btab = dict(IEC._B_WEIGHT)
    rtab = dict(IEC._R_WEIGHT)
    badb = sum(abs(btab[lam] - b) > 1e-9 for lam, (b, _) in rows.items()
               if lam in btab)
    badr = sum(abs(rtab[lam] - r) > 1e-9 for lam, (_, r) in rows.items()
               if lam >= 380 and lam in rtab)
    check("B4  code _B_WEIGHT/_R_WEIGHT == text knots (inverse check)",
          badb == 0 and badr == 0, "B:%d R:%d" % (badb, badr))
    # analytic tail bands vs code branch formulas
    def bf(lam):  # text B tail formulas
        if 500 < lam <= 600:
            return 10.0 ** ((450.0 - lam) / 50.0)
        if 600 < lam <= 700:
            return 0.001
        return None

    def rf(lam):  # text R tail formulas
        if 500 < lam <= 700:
            return 1.0
        if 700 < lam <= 1050:
            return 10.0 ** ((700.0 - lam) / 500.0)
        if 1050 < lam <= 1150:
            return 0.2
        if 1150 < lam <= 1200:
            return 0.2 * 10.0 ** (0.02 * (1150.0 - lam))
        if 1200 < lam <= 1400:
            return 0.02
        return None

    for lam, fn, name in [(550, bf, "B 500-600"), (650, bf, "B 600-700"),
                          (600, rf, "R 500-700"), (800, rf, "R 700-1050"),
                          (1100, rf, "R 1050-1150"), (1175, rf, "R 1150-1200"),
                          (1300, rf, "R 1200-1400")]:
        codev = float(IEC._b_weight(lam) if name.startswith("B") else IEC._r_weight(lam))
        want = fn(lam)
        check("B5  %s: code %s == text formula %s" % (name, codev, want),
              abs(codev - want) < 1e-9)
    needle("B6  text band B(550 nm) = 10^((450-lambda)/50)", "10^((450-λ)/50)")
    needle("B7  text band R(700-1050) = 10^((700-lambda)/500)", "10^((700-λ)/500)")
    needle("B8  text band R(1150-1200) = 0.2*10^(0.02*(1150-lambda))",
           "0.2*10^(0.02*(1150-λ))")
    info("B9  iec62471.py _B_WEIGHT carries a boundary knot at 500 nm that is "
         "NOT printed in this edition's Table 4.2 (verified on the actual PDF "
         "page layout: the table jumps from row 495 straight to the '500-600' "
         "band row). The band formula gives B(500) = 10^((450-500)/50) = 0.1, "
         "so the code knot is aligned to the printed curve value (0.1), keeping "
         "interpolation continuous with the 500 < lambda <= 600 branch.")
    check("B10  code B(500) boundary knot == band formula value 0.1",
          abs(float(IEC._b_weight(500.0)) - 0.1) < 1e-9,
          "B(500)=%.6f" % float(IEC._b_weight(500.0)))


# --------------------------------------------------------------------------
# [C] Table 4.1 S_UV(lambda) -- reference-only -------------------------------
# --------------------------------------------------------------------------
def check_table41():
    section("[C] Table 4.1 S_UV(lambda) (reference-only transcription guard)")
    sec, ok_b = slice_between("Table 4.1 Spectral weighting function",
                              "4.3.2 Near-UV hazard exposure limit")
    check("C1  Table 4.1 region delimited", ok_b)
    toks = numbers(sec)[:108]  # 54 (lambda, value) pairs; drop trailing residue
    pairs = [(int(w[0]), float(w[1])) for w in zip(toks[0::2], toks[1::2])]
    left = dict(pairs[0::2])   # wavelengths 200-310 (27 rows)
    right = dict(pairs[1::2])  # wavelengths 313-400 (27 rows)
    want_left = {200: 0.030, 205: 0.051, 210: 0.075, 215: 0.095, 220: 0.120,
                 225: 0.150, 230: 0.190, 235: 0.240, 240: 0.300, 245: 0.360,
                 250: 0.430, 254: 0.500, 255: 0.520, 260: 0.650, 265: 0.810,
                 270: 1.000, 275: 0.960, 280: 0.880, 285: 0.770, 290: 0.640,
                 295: 0.540, 297: 0.460, 300: 0.300, 303: 0.120, 305: 0.060,
                 308: 0.026, 310: 0.015}
    want_right = {313: 0.006, 315: 0.003, 316: 0.0024, 317: 0.0020,
                  318: 0.0016, 319: 0.0012, 320: 0.0010, 322: 0.00067,
                  323: 0.00054, 325: 0.00050, 328: 0.00044, 330: 0.00041,
                  333: 0.00037, 335: 0.00034, 340: 0.00028, 345: 0.00024,
                  350: 0.00020, 355: 0.00016, 360: 0.00013, 365: 0.00011,
                  370: 0.000093, 375: 0.000077, 380: 0.000064, 385: 0.000053,
                  390: 0.000044, 395: 0.000036, 400: 0.000030}
    ok_c1 = (len(left) == 27 and len(right) == 27)
    ok_c2 = (left == want_left)
    ok_c3 = (right == want_right)
    check("C2  27+27 wavelength/value rows de-interleaved", ok_c1,
          "left=%d right=%d" % (len(left), len(right)))
    check("C3  left column (200-310 nm) matches IEC Table 4.1", ok_c2,
          "peak at 270 nm = %s" % left.get(270))
    check("C4  right column (313-400 nm) matches IEC Table 4.1", ok_c3,
          "tail at 400 nm = %s" % right.get(400))
    check("C5  footnote: values obtained by logarithmic interpolation",
          "logarithmic interpolation" in text)
    check("C6  footnote: emission lines of a mercury discharge spectrum",
          "mercury discharge spectrum" in text)

    info("C7  code _uv_weight() uses the Figure 4.1 BOX approximation "
         "(S_UV=1 for 200-315 nm, 10^(0.02*(315-lambda)) after), NOT these "
         "Table 4.1 discrete values. kS is supplementary-only and the "
         "actinic clause is acknowledged out-of-scope for the sun."
         " See (C7).")


# --------------------------------------------------------------------------
# [D] Table 5.4 irradiance ELs ----------------------------------------------
# --------------------------------------------------------------------------
def check_table54():
    section("[D] Table 5.4 irradiance ELs (skin/cornea)")
    needle("D1  actinic UV: constant irradiance 30/t", "30/t")
    needle("D2  actinic UV: whole-body < 30000 s", "<30000")
    needle("D3  eye UV-A: 10000/t (<=1000 s) then 10 W/m^2", "10000/t")
    needle("D4  eye UV-A floor 10 W/m^2", "10")
    needle("D5  blue small source: 100/t (<=100 s) then 1.0 W/m^2", "100/t")
    needle("D6  blue small source floor 1.0 W/m^2", "1.0")
    needle("D7  eye IR: 18000/t^0.75 (<=1000 s) then 100 W/m^2",
           "18000/t^0.75")
    needle("D8  eye IR floor 100 W/m^2", "100")
    needle("D9  skin thermal: 20000/t^0.75 (<=10 s) then 100 W/m^2",
           "20000/t^0.75")
    needle("D10 limiting aperture 1.4 (80 deg)", "1.4(80)")
    needle("D11 skin thermal field 2*pi sr", "2πsr")


# --------------------------------------------------------------------------
# [E] Equations 4.1-4.11 verbatim -------------------------------------------
# --------------------------------------------------------------------------
def check_eqs():
    section("[E] Equations 4.1-4.11 (constants as written)")
    needle("E1  (4.1) actinic: <= 30 J/m^2", "30J/m^2")
    needle("E2  (4.2) t_max = 30/E_s", "t_max=30/E_s")
    needle("E3  (4.3a) UV-A: <= 10000 J/m^2 (t<1000 s)", "10000J/m^2")
    needle("E4  (4.3b) UV-A: E_UVA <= 10 W/m^2 (t>=1000 s)", "E_UVA≤10W/m^2")
    needle("E5  (4.4) t_max <= 10000/E_UVA", "t_max≤10000/E_UVA")
    needle("E6  (4.6) t_max = 10^6/L_B (t<=10^4 s)", "t_max=10^6/L_B")
    needle("E7  (4.7a) blue small: <= 100 J/m^2 (t<=100 s)", "100J/m^2")
    needle("E8  (4.7b) blue small: <= 1 W/m^2 (t>100 s)", "≤1W/m^2")
    needle("E9  (4.8) t_max = 100/E_B", "t_max=100/E_B")
    needle("E10 (4.9) thermal EL: 50000/(alpha*t^0.25) W/(m^2*sr)",
           "50000/(α*t^0.25)W/(m^2*sr)")
    needle("E11 (4.9) validity window 10 us <= t <= 10 s", "10µs≤t≤10s")
    needle("E12 (4.10) weak-stimulus: 6000/alpha W/(m^2*sr)", "6000/αW/(m^2*sr)")
    needle("E13 (4.10) weak stimulus < 10 cd/m^2", "less than 10 cd/m^2")
    needle("E14 3mm-pupil scale-up 33000/alpha (clause 4.2.1)", "33000/α")
    needle("E15 (4.11) eye IR: 18000/(t^0.75) W/m^2 (t<1000 s)",
           "18000/(t^0.75)W/m^2")


# --------------------------------------------------------------------------
# [F] Table 6.1 risk-group emission limits ----------------------------------
# --------------------------------------------------------------------------
def check_table61():
    section("[F] Table 6.1 risk-group emission limits")
    sec, ok_b = slice_between("Table 6.1 Emission limits",
                              "SUMMARY OF BIOLOGICAL EFFECTS")
    check("F0  Table 6.1 region delimited", ok_b)
    n = [float(t) for t in numbers(sec)] if ok_b else []
    need = {0.001, 0.003, 0.03, 10, 33, 100, 10000, 4000000, 1.0, 400,
            570, 3200}
    missing = sorted(need - set(n)) if ok_b else need
    check("F1  all numeric emission-limit values present", not missing,
          "missing=%s" % missing)
    needle("F2  retinal thermal exempt = 28000/alpha (Exempt == RG1)",
           "28000/α")
    needle("F3  retinal thermal RG2 = 71000/alpha", "71000/α")
    needle("F4  weak-stimulus L_IR = 6000/alpha (x3 ranges)", "6000/α")
    needle("F5  small-source footnote: alpha < 0.011 radian, FOV 0.1 rad at "
           "10000 s", "α<0.011radian")
    needle("F6  weak-stimulus footnote: non-GLS source",
           "evaluation of non-GLS source")
    needle("F7  units W/m^2 and W/(m^2*sr) present", "W/(m^2*sr)")
    check("F8  code UV limits == Table 6.1 (0.001/0.003/0.03)",
          (IEC.UV_EXEMPT, IEC.UV_RG1, IEC.UV_RG2) == (0.001, 0.003, 0.03),
          (IEC.UV_EXEMPT, IEC.UV_RG1, IEC.UV_RG2))
    check("F9  code UVA limits == Table 6.1 (10/33/100)",
          (IEC.UVA_EXEMPT, IEC.UVA_RG1, IEC.UVA_RG2) == (10.0, 33.0, 100.0),
          (IEC.UVA_EXEMPT, IEC.UVA_RG1, IEC.UVA_RG2))
    check("F10 code blue radiance limits == Table 6.1 (100/1e4/4e6)",
          (IEC.BLUE_EXEMPT_RAD, IEC.BLUE_RG1_RAD, IEC.BLUE_RG2_RAD)
          == (100.0, 1.0e4, 4.0e6),
          (IEC.BLUE_EXEMPT_RAD, IEC.BLUE_RG1_RAD, IEC.BLUE_RG2_RAD))
    check("F11 code blue small-source limits == Table 6.1 (1.0/1.0/400)",
          (IEC.BLUE_EXEMPT_E, IEC.BLUE_RG1_E, IEC.BLUE_RG2_E)
          == (1.0, 1.0, 400.0),
          (IEC.BLUE_EXEMPT_E, IEC.BLUE_RG1_E, IEC.BLUE_RG2_E))
    check("F12 code IR eye limits == Table 6.1 (100/570/3200)",
          (IEC.IR_EXEMPT, IEC.IR_RG1, IEC.IR_RG2) == (100.0, 570.0, 3200.0),
          (IEC.IR_EXEMPT, IEC.IR_RG1, IEC.IR_RG2))
    check("F13 code thermal constants 28000/28000/71000 == Table 6.1",
          IEC._THERMAL_EXEMPT_K == 28000.0 and IEC._THERMAL_RG2_K == 71000.0,
          "%r/%r" % (IEC._THERMAL_EXEMPT_K, IEC._THERMAL_RG2_K))


# --------------------------------------------------------------------------
# [G] Table 5.5 radiance ELs ------------------------------------------------
# --------------------------------------------------------------------------
def check_table55():
    section("[G] Table 5.5 radiance ELs (retina)")
    needle("G1  blue L_B: 10^6/t (0.25-10 s, 10-100 s, 100-10000 s)",
           "10^6/t")
    needle("G2  blue L_B: 100 W/(m^2*sr) (t>=10000 s)", "100")
    needle("G3  retinal thermal: 50000/(alpha*t^0.25) (t<0.25 s, 0.25-10 s)",
           "50000/(α*t^0.25)")
    needle("G4  thermal FOV: t<0.25 s uses 0.0017 rad cone", "0.0017")
    needle("G5  thermal FOV ramp: 0.011*sqrt(t/10)", "0.011*sqrt(t/10)")
    needle("G6  blue FOV ramp (t>10000 s? 100-10000 s): 0.0011*sqrt(t)",
           "0.0011*sqrt(t)")
    needle("G7  weak-stimulus: 6000/alpha (t>10 s)", "6000/α")
    needle("G8  blue FOV floor 0.011 rad (10-100 s)", "0.011")
    needle("G9  blue FOV cap 0.1 rad (t>=10000 s)", "0.1")


# --------------------------------------------------------------------------
# [H] code self-consistency re-derived from the text formulas ----------------
# --------------------------------------------------------------------------
def check_code_consistency():
    section("[H] code numbers re-derived from the text formulas")
    # Table 6.1 thermal limits vs Eq 4.9 evaluated at the group windows
    # (Exempt/RG1 uses t=10 s, RG2 uses t=0.25 s blink):
    r10 = 5.0e4 / 10.0 ** 0.25
    r025 = 5.0e4 / 0.25 ** 0.25
    check("H1  exempt 2.8e4/alpha == Eq 4.9 at t=10 s (within 1%%)",
          abs(IEC._THERMAL_EXEMPT_K - r10) / r10 < 0.01,
          "2.8e4 vs %.0f" % r10)
    check("H2  RG2 7.1e4/alpha == Eq 4.9 at t=0.25 s (within 1%%)",
          abs(IEC._THERMAL_RG2_K - r025) / r025 < 0.01,
          "7.1e4 vs %.0f" % r025)
    # fov_gamma vs the text's alpha_eff(t) (Sec 4.2.2 / Fig 5.4-5.5)
    g = IEC.fov_gamma(np.array([9.0, 100.0, 1000.0, 5000.0, 1.0e5]))
    want_g = [0.011, 0.011, 0.011 * (10.0 ** 0.5),
              0.011 * (50.0 ** 0.5), 0.1]
    ok_g = np.allclose(g, want_g, rtol=1e-9)
    check("H3  fov_gamma(t) == 0.011 / 0.011 / 0.011*sqrt(t/100) / 0.1",
          bool(ok_g), ["%.6e" % x for x in g])
    # EL functions re-derived from Eqs 4.6/4.8/4.9
    check("H4  blue_light_el(10) = 1e6/10 = 1e5", 
          abs(IEC.blue_light_el(10.0) - 1.0e5) < 1e-6)
    check("H5  blue_light_el(5e4) = 100 (t>1e4 s)",
          abs(IEC.blue_light_el(5.0e4) - 100.0) < 1e-6)
    check("H6  blue_light_small_el(50) = 100/50 = 2",
          abs(IEC.blue_light_small_el(50.0) - 2.0) < 1e-9)
    check("H7  retinal_thermal_el: 5e4/(alpha*t^0.25) at alpha=0.1,t=0.25",
          abs(IEC.retinal_thermal_el(0.1, 0.25)
              - 5.0e4 / (0.1 * 0.25 ** 0.25)) < 1e-6)
    # IEC small-source blue decision tree threshold alpha < 0.011 (4.3.4)
    check("H8  ALPHA_SMALL == blue FOV 0.011 (t<=100 s)",
          abs(IEC.ALPHA_SMALL - 0.011) < 1e-12)


# --------------------------------------------------------------------------
# [J] PD IEC/TR 62471-2:2009 cross-check (edition-note provenance) ----------
# --------------------------------------------------------------------------
def check_tr2():
    section("[J] PD IEC/TR 62471-2:2009 cross-check (guidance only)")
    tr_txt = os.path.join(REPO, "references", "_PD_IEC_TR_62471_2_2009_lock.txt")
    check("J1  TR extract present (references/_PD_IEC_TR_62471_2_2009_lock.txt)",
          os.path.isfile(tr_txt), tr_txt)
    if not os.path.isfile(tr_txt):
        return
    tt = open(tr_txt, encoding="utf-8").read()
    tflat = re.sub(r"\s+", "", tt)
    needle = lambda name, s: check(name, re.sub(r"\s+", "", s) in tflat,
                                   ("needle '%s' found" % s) if
                                   re.sub(r"\s+", "", s) in tflat else
                                   ("needle '%s' NOT FOUND" % s))
    needle("J2  TR is PD IEC/TR 62471-2:2009", "PD IEC/TR 62471-2:2009")
    needle("J3  TR is manufacturer guidance (hazard distance), not a limit table",
           "Determination of hazard distances")
    needle("J4  TR keeps alpha acceptance 1.7 mrad minimum", "1.7 mrad")
    needle("J5  TR keeps blue Exempt 100 mrad acceptance", "100 mrad")
    needle("J6  TR keeps blue Exempt radiance EL 100 W/(m^2*sr)",
           "100 W/(m^2*sr)")
    # Provenance guard on the docstring correction: an earlier note claimed PD
    # IEC/TR 62471-2:2009 discusses a 0.01 W/m2 small-source alternative; the
    # stored TR does NOT contain that value as a unit-bearing limit.
    bad_unit = re.findall(r"0\.0?1\s*W/?m2|0\.0?1\s*W/m\^2|0,0?1\s*W/?m2",
                          re.sub(r"\s+", " ", tt))
    bad_plain = [ln for ln in tt.splitlines()
                 if ln.strip() in ("0.01", "0,01")]  # axis ticks only, no unit
    check("J7  TR does NOT state a 0.01 W/m^2 blue-light limit",
          not bad_unit,
          ("hits: %r" % bad_unit) if bad_unit else "no unit-bearing 0.01 W/m2")
    check("J8  only log-axis ticks carry the bare 0.01 (unit-less)",
          len(bad_plain) >= 2,
          "bare-tick lines %d (>=2 expected)" % len(bad_plain))
    info("J9  PD IEC/TR 62471-2:2009 is manufacturer guidance built on IEC "
         "62471:2006's limit table; it redefines no exposure limit, so it does "
         "not affect the code's Table 6.1 values (blue small-source Exempt = "
         "1.0 W/m2 per BS EN 62471:2008). Annex B hazard distances are not "
         "implemented in the pipeline (the analysis classifies each receptor "
         "at its fixed position).")


# --------------------------------------------------------------------------
# [I] INFO notes / documented simplifications --------------------------------
# --------------------------------------------------------------------------
def info_notes():
    section("[I] INFO: documented simplifications (not pass/fail)")
    info("I1  FOV gamma collapsed to 0.011 rad for ALL t <= 100 s; the text "
         "only ramps alpha_min*sqrt(t/0.25) over 0.25-10 s. Provably benign "
         "on the current config (alpha ~ 0.05 rad > gamma, so dilution "
         "saturates at 1); would matter only for alpha < 0.011 rad sources.")
    info("I2  weak-visual-stimulus L_IR 6000/alpha (non-GLS, Table 6.1/Eq "
         "4.10) is not implemented as a separate clause - the code uses the "
         "R(lambda) thermal radiance for the FORMAL eye group only. Consistent "
         "with the footnote '** Involves evaluation of non-GLS source'.")
    w = IEC.spectral_weights()
    info("I3  k_R=%.5f, k_B=%.5f are COMPUTED (IEC 60904-3:2019 Table 1 "
         "direct x Table 4.2, "
         "full-spectrum normalisation), not transcribed from the standard "
         "text; inputs live in the text, the numbers in validate_iec62471.py "
         "(AGENTS.md pins k_R=1.290, k_B=0.0859)." % (w["kR"], w["kB"]))


# --------------------------------------------------------------------------
def main():
    global text, flat
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("validate_norm_vs_code.py: normalized text vs iec62471.py\n")
        f.write("=" * 70 + "\n")

    if not os.path.isfile(TXT):
        check("file exists: references/_BS_EN_62471_2008.txt",
              os.path.isfile(TXT), TXT)
        return 1
    text = open(TXT, encoding="utf-8").read()
    flat = re.sub(r"\s+", "", text)

    check_angular()
    check_table42()
    check_table41()
    check_table54()
    check_eqs()
    check_table61()
    check_table55()
    check_code_consistency()
    check_tr2()
    info_notes()

    summary = "\nSummary: %d passed, %d failed, %d info\n" % (PASS, FAIL, INF)
    print(summary)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(summary)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())