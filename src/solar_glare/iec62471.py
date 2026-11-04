"""IEC 62471 photobiological assessment of solar glare (BS EN 62471:2008).

Secondary hazard method that runs alongside the Sandia SGHAT classification
(see `hazard.py`). It maps the already-computed corneal irradiance E_c and
retinal irradiance E_r of the glare minutes onto the IEC 62471 exposure-limit
(EL) framework WITHOUT re-integrating a spectrum per minute:

  * Source radiance of the reflected beam. A flat mirror preserves the solar
    radiance (no geometric concentration), so
        L = rho * DNI / Omega_sun
    which, using the pipeline's E_c = rho * DNI (W/m2), re-expresses as
        L = E_c * 1e4 / Omega_sun                      [W m-2 sr-1]
    where Omega_sun = 2*pi*(1 - cos(sun_disc/2)) ~ 6.792e-5 sr.
    Because this comes from E_c directly it is independent of the SGHAT
    sub-array footprint / subtended angle (radiance is invariant with
    observer distance for a specular reflector).

  * Spectral weighting k_X, computed ONCE against a direct-normal AM1.5
    reference spectrum (IEC 60904-3:2019 Table 1 Direct, = ASTM G173-03,
    loaded from src/solar_glare/data/IEC60904-3_2019_Table1.csv):
        k_X = sum(S(lam) W_X(lam) dlam) / sum(S(lam) dlam)
    assuming a spectrally-flat panel reflectivity rho(lam) ~ const.
    Then  E_X = k_X * E_c   and   L_X = k_X * L.
    Weighted irradiances/redefined below follow the clause action functions
    of the standard: S_UV (actinic), B (blue light), R (retinal thermal).

Hazards evaluated (clause / Table 6.1 emission limits):
  * 4.3.5 retinal thermal (L_R, radial dependence)  -- FORMAL eye risk group
  * 4.3.3/4.3.4 blue light (L_B / small-source E_B) -- FORMAL eye risk group
  * 4.3.1 actinic UV (E_S), 4.3.2 near-UV (E_UVA), 4.3.7 IR (E_IR),
    4.3.8 skin (E_H)                               -- SUPPLEMENTARY only.

FOV averaging / dilution (Sec 4.2.2, 5.2-5.4): the radiance to be compared
with the limits is the average over a measurement cone gamma = alpha_eff(t):
    gamma = 0.011 rad           for t <= 100 s
    gamma = 0.011*sqrt(t/100)   for 100 s < t < 1e4 s
    gamma = 0.1 rad             for t >= 1e4 s
For a source subtending less than gamma, the averaged radiance is diluted by
the area ratio (alpha/gamma)^2. The dilution only ever LOWERS L_avg, so it is
conservative to ignore it; we still compute it for the reported reference
duration so the Exempt verdict is correct.

Blue-light decision tree (Sec 4.2.2): alpha < 0.011 rad -> small-source
irradiance E_B (Sec 4.3.4, no dilution); alpha >= 0.011 rad -> radiance L_B
(Sec 4.3.3, with FOV dilution).

Risk group assignment: the FORMAL group is the worst of the eye clauses
(blue light + retinal thermal). The UV/UVA/IR/skin clauses are reported
separately/ incrementally because the actinic-UV Exempt limit is so tight that
ordinary direct sunlight exceeds it by ~2 decades (a well-known scope caveat:
IEC 62471 targets lamps, not the sun) - quoting the UV clause as the culprit
would mislead.

Edition note: values follow BS EN 62471:2008 (= IEC 62471:2006 with CENELEC
common modifications). Table 6.1 of that edition gives the blue-light
small-source Exempt limit as 1.0 W/m2 (footnote "*"): "Small source defined
as one with alpha < 0.011 radian. Averaging field of view at 10000 s is
0.1 radian."  An earlier note here cited PD IEC/TR 62471-2:2009 as
discussing a 0.01 W/m2 alternative; a 2026-09-20 re-read of the stored TR
extract (references/_PD_IEC_TR_62471_2_2009_lock.txt) found NO such value
anywhere in it - that TR is manufacturer guidance only (hazard distances,
acceptance angles: min 1.7 mrad, blue 100 mrad Exempt / 11 mrad RG1-2;
labelling), built on the same IEC 62471:2006 limit table, and does not
redefine any exposure limit. The code uses the 1.0 W/m2 of BS EN 62471:2008
Table 6.1. State the edition used in any report.
"""

import os

import numpy as np

from .profiling import stage_fn

# --- Optical / source constants -------------------------------------------------
# Matched to hazard.py so the IEC and SGHAT numbers share one eye model.
PUPIL_DIAMETER = 0.002       # m (daylight-adapted)
EYE_FOCAL_LENGTH = 0.017     # m
OCULAR_TRANSMISSION = 0.5    # ocular media transmission
SUN_SUBTENDED_RAD = 9.3e-3   # apparent sun diameter (rad)

# Solid angle subtended by the sun disc (full formula, SGHAT v6 uses the
# pi*beta^2/4 small-angle approximation; the exact value is 1.0003x it).
OMEGA_SUN = 2.0 * np.pi * (1.0 - np.cos(SUN_SUBTENDED_RAD / 2.0))

# Optics factor linking source radiance L [W m-2 sr-1] to retinal irradiance
# E_r [W m-2] for an extended source:  E_r = L * (pi/4) * (dp/f)^2 * tau.
K_EYE = (np.pi / 4.0) * (PUPIL_DIAMETER / EYE_FOCAL_LENGTH) ** 2 * OCULAR_TRANSMISSION

# --- IEC 62471 angular-definition constants (Sec 4.2.2) -------------------------
ALPHA_MIN = 0.0017           # rad, smallest retinal image (point source)
ALPHA_MAX = 0.1              # rad, largest treated source (independent above)
ALPHA_SMALL = 0.011          # rad, small-source threshold = blue FOV for t<=100s

# Exposure durations used by the risk-group definition (Sec 6).
T_EXEMPT = 1.0e4             # s  (blue light / UVA / UV exempt window)
T_RG1 = 100.0               # s  (blue light RG1 window)
T_AVERSION = 0.25           # s  (aversion / blink response, RG2 window)
T_RG1_THERMAL = 10.0        # s  (retinal thermal exempt/low window, Table 6.1)

# --- Table 6.1 emission limits (BS EN 62471:2008) ------------------------------
# Each tuple is (Exempt, RG1, RG2); a value above the RG2 limit is RG3.

# Retinal thermal, radiance form [W m-2 sr-1]: exempt == RG1 both 2.8e4/alpha.
# (The underlying EL equation, Sec 4.3.5, is 5e4/(alpha*t^0.25); at t=10 s that
# is 2.8117e4/alpha, rounded to 2.8e4 in Table 6.1. We use the TABLE values for
# the group boundaries and the EQUATION form for the t_max inversion.)
_THERMAL_EXEMPT_K = 28000.0
_THERMAL_RG2_K = 71000.0
# Thermal EL from Eq 4.9: EL = 5e4/(alpha * t^0.25). Inverted -> t_max.
THERMAL_EL_K = 50000.0

# Blue light, radiance form [W m-2 sr-1] (Sec 4.3.3 / Table 6.1).
(BLUE_EXEMPT_RAD, BLUE_RG1_RAD, BLUE_RG2_RAD) = (100.0, 1.0e4, 4.0e6)
# Blue light small-source irradiance form [W m-2] (Sec 4.3.4 / Table 6.1).
(BLUE_EXEMPT_E, BLUE_RG1_E, BLUE_RG2_E) = (1.0, 1.0, 400.0)

# Supplementary (informational) limits from Table 6.1 [W m-2].
(UV_EXEMPT, UV_RG1, UV_RG2) = (0.001, 0.003, 0.03)      # actinic UV E_S
(UVA_EXEMPT, UVA_RG1, UVA_RG2) = (10.0, 33.0, 100.0)    # near-UV E_UVA
(IR_EXEMPT, IR_RG1, IR_RG2) = (100.0, 570.0, 3200.0)    # IR eye E_IR

# --- Spectral weight computation (AM1.5 direct normal, once) --------------------
_B_WEIGHT = [  # Table 4.2 blue-light hazard weighting B(lambda), 300-500 nm.
    (300.0, 0.01), (355.0, 0.01), (360.0, 0.01), (365.0, 0.01), (370.0, 0.01),
    (375.0, 0.01), (380.0, 0.01), (385.0, 0.013), (390.0, 0.025), (395.0, 0.05),
    (400.0, 0.10), (405.0, 0.20), (410.0, 0.40), (415.0, 0.80), (420.0, 0.90),
    (425.0, 0.95), (430.0, 0.98), (435.0, 1.00), (440.0, 1.00), (445.0, 0.97),
    (450.0, 0.94), (455.0, 0.90), (460.0, 0.80), (465.0, 0.70), (470.0, 0.62),
    (475.0, 0.55), (480.0, 0.45), (485.0, 0.40), (490.0, 0.22), (495.0, 0.16),
    # Table 4.2 prints rows only to 495; the "500-600" band row gives
    # B = 10^((450-lambda)/50) = 0.1 at 500. No standalone 500 knot is
    # printed, so the boundary knot is the band value (keeps interpolation
    # continuous with the 500 < λ <= 600 branch in _b_weight).
    (500.0, 0.1),
]

_R_WEIGHT = [  # Table 4.2 burn (retinal thermal) weighting R(lambda), 380-500 nm.
    (380.0, 0.1), (385.0, 0.13), (390.0, 0.25), (395.0, 0.5),
    (400.0, 1.0), (405.0, 2.0), (410.0, 4.0), (415.0, 8.0),
    (420.0, 9.0), (425.0, 9.5), (430.0, 9.8), (435.0, 10.0),
    (440.0, 10.0), (445.0, 9.7), (450.0, 9.4), (455.0, 9.0),
    (460.0, 8.0), (465.0, 7.0), (470.0, 6.2), (475.0, 5.5),
    (480.0, 4.5), (485.0, 4.0), (490.0, 2.2), (495.0, 1.6),
    (500.0, 1.0),
]

_SPECTRAL_WEIGHTS = None  # module-level cache {name: k}

_SPECTRUM_CACHE = None


def _spectrum():
    """Direct-normal AM1.5 reference spectrum (IEC 60904-3:2019 Table 1).

    Returns (wavelength_nm, irradiance [W/m^2/nm]) from the Direct column of
    IEC 60904-3:2019 Table 1 (BS EN IEC 60904-3:2019, 2002 points, 280-4000 nm,
    SMARTS 2.9.2; the standard states these values equal ASTM G173-03). The
    table is committed as ``src/solar_glare/data/IEC60904-3_2019_Table1.csv``
    (regenerated from the reference extract by
    ``scripts/export_iec60904_3_spectrum.py``).
    """
    global _SPECTRUM_CACHE
    if _SPECTRUM_CACHE is None:
        import numpy as _np

        path = os.path.join(os.path.dirname(__file__), "data",
                            "IEC60904-3_2019_Table1.csv")
        data = _np.genfromtxt(path, delimiter=",", skip_header=1, encoding="utf-8")
        _SPECTRUM_CACHE = (data[:, 0], data[:, 4])  # wavelength, direct column
    return _SPECTRUM_CACHE


def reference_spectrum_dni():
    """IEC 60904-3:2019 Table 1 direct-normal reference spectrum (wl_nm, E).

    Public alias of ``_spectrum``, shared with
    ``hazard.solar_weighted_reflectivity`` (E903-20-style solar weighting of
    the panel reflectance over the same IEC 60904-3 spectrum used here).
    """
    return _spectrum()


def _b_weight(wl_nm):
    """Blue-light hazard weighting B(lambda), Table 4.2 (linear interp)."""
    if wl_nm < 300.0 or wl_nm > 700.0:
        return 0.0
    if 500.0 < wl_nm <= 600.0:
        return 10.0 ** ((450.0 - wl_nm) / 50.0)
    if 600.0 < wl_nm <= 700.0:
        return 0.001
    xs = [p[0] for p in _B_WEIGHT]
    ys = [p[1] for p in _B_WEIGHT]
    j = max(int(np.searchsorted(xs, wl_nm)) - 1, 0)
    a, b = xs[j], xs[j + 1]
    return ys[j] + (ys[j + 1] - ys[j]) * (wl_nm - a) / (b - a)


def _r_weight(wl_nm):
    """Burn (retinal thermal) weighting R(lambda), Table 4.2.

    380-500 nm: tabulated blue-violet hump (peak 10.0 at 435-440), log-interp;
    500-700 nm: flat 1.0; then the near-IR power-law tail down to 0.02.
    """
    if wl_nm < 380.0 or wl_nm >= 1400.0:
        return 0.0
    if 380.0 <= wl_nm < 500.0:
        xs = np.array([p[0] for p in _R_WEIGHT])[:-1]
        ys = np.log10(np.array([p[1] for p in _R_WEIGHT])[:-1])
        return 10.0 ** float(np.interp(wl_nm, xs, ys))
    if 500.0 <= wl_nm < 700.0:
        return 1.0
    if 700.0 <= wl_nm < 1050.0:
        return 10.0 ** ((700.0 - wl_nm) / 500.0)
    if 1050.0 <= wl_nm < 1150.0:
        return 0.2
    if 1150.0 <= wl_nm < 1200.0:
        return 0.2 * 10.0 ** (0.02 * (1150.0 - wl_nm))
    if 1200.0 <= wl_nm < 1400.0:
        return 0.02
    return 0.0


def _uv_weight(wl_nm):
    """Actinic UV (skin/eye) weighting S_UV, Table 4.1 box profile."""
    if 200.0 <= wl_nm < 315.0:
        return 1.0
    if 315.0 <= wl_nm < 400.0:
        return 10.0 ** (0.02 * (315.0 - wl_nm))
    return 0.0


def _uva_weight(wl_nm):
    """Near-UV weighting (315 < lambda <= 400 nm, boxcar 1.0)."""
    return 1.0 if 315.0 < wl_nm <= 400.0 else 0.0


def _ir_weight(wl_nm):
    """IR radiation hazard (cornea/lens), 780-3000 nm boxcar (1.0)."""
    return 1.0 if 780.0 <= wl_nm <= 3000.0 else 0.0


def _skin_weight(wl_nm):
    """Skin thermal hazard, 380-3000 nm boxcar (1.0)."""
    return 1.0 if 380.0 <= wl_nm <= 3000.0 else 0.0


def spectral_weights():
    """Spectral weighting factors k_X for the sun spectrum (once, cached).

    k_X = sum(S W_X dlam) / sum(S dlam) with BOTH integrals over the FULL
    spectrum band (200-4000 nm), because E_c is the broadband irradiance:
    E_X = E_c * (integral S W_X / integral S_full). Assumes a spectrally-flat
    panel reflectivity. Returns {kS, kUVA, kB, kR, kIR, kH}.
    """
    global _SPECTRAL_WEIGHTS
    if _SPECTRAL_WEIGHTS is not None:
        return _SPECTRAL_WEIGHTS
    wl, e = _spectrum()
    # IEC 60904-3:2019 Table 1 starts at 280 nm; the <280 nm solar
    # contribution is negligible at ground level but padded so kS integrates
    # the full S_UV band.
    if wl[0] > 200.0:
        wl = np.concatenate([[200.0], wl])
        e = np.concatenate([[0.0], e])
    full = np.argsort(wl)
    wl = wl[full]
    e = e[full]
    denom = float(np.trapezoid(e, wl))  # full-spectrum (200-4000 nm) irradiance

    def _k(mask, wfunc):
        wm = e[mask] * np.array([wfunc(x) for x in wl[mask]])
        return float(np.trapezoid(wm, wl[mask])) / denom

    kS = _k((wl >= 200) & (wl < 400), _uv_weight)
    kUVA = _k((wl > 315) & (wl <= 400), _uva_weight)
    kB = _k((wl >= 300) & (wl <= 700), _b_weight)
    kR = _k((wl >= 380) & (wl < 1400), _r_weight)
    kIR = _k((wl >= 780) & (wl <= 3000), _ir_weight)
    kH = _k((wl >= 380) & (wl <= 3000), _skin_weight)
    _SPECTRAL_WEIGHTS = {"kS": kS, "kUVA": kUVA, "kB": kB, "kR": kR, "kIR": kIR, "kH": kH}
    return _SPECTRAL_WEIGHTS


# --- Continuum functions --------------------------------------------------------
def solar_disc_solid_angle():
    """Solid angle of the sun disc (sr)."""
    return OMEGA_SUN


def source_radiance(ec_wcm2):
    """Reflected-beam radiance L [W m-2 sr-1] from corneal irradiance.

    L = E_c(W/m2) / Omega_sun = ec_wcm2 * 1e4 / Omega_sun.

    REF: radiance conservation for a flat specular reflector (no concentration);
    new5/guide and SGHAT-v6 Sec 4.3 use the same 1.47e4 L/E_c factor at the
    9.3 mrad sun disc.
    """
    return np.asarray(ec_wcm2, dtype=float) * 1e4 / OMEGA_SUN


def angular_subtense(width_m, height_m, distance_m, abs_cos_ref=1.0):
    """Apparent source angular subtense alpha (rad), IEC 62471 Sec 4.2.2.

    alpha = arithmetic mean of the clamped max/min angular dimensions of the
    panel rectangle: each dimension is bounded to [ALPHA_MIN, ALPHA_MAX]
    BEFORE the mean. The width (tilt-plane) dimension is foreshortened by
    abs_cos_ref (angle between reflected ray and panel normal), matching the
    SGHAT v6 Sec 5.5 visibility factor.
    """
    d = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    w = np.maximum(np.asarray(width_m, dtype=float) * np.maximum(abs_cos_ref, 0.0), 0.0)
    h = np.maximum(np.asarray(height_m, dtype=float), 0.0)
    a_w = np.clip(w / d, ALPHA_MIN, ALPHA_MAX)
    a_h = np.clip(h / d, ALPHA_MIN, ALPHA_MAX)
    return (a_w + a_h) / 2.0


def fov_gamma(t_sec):
    """Averaging field-of-view cone gamma = alpha_eff(t) (rad), Sec 4.2.2."""
    t = np.maximum(np.asarray(t_sec, dtype=float), 1e-9)
    g = np.empty_like(t)
    low = t <= 100.0
    high = t >= 1.0e4
    g[low] = ALPHA_SMALL
    g[high] = ALPHA_MAX
    mid = (~low) & (~high)
    g[mid] = ALPHA_SMALL * np.sqrt(t[mid] / 100.0)
    return g


def fov_dilution(alpha_rad, t_sec):
    """FOV-averaging dilution factor (<=1) for radiance measurements.

    For a source subtending less than the measurement cone gamma(t) the
    radiance averaged over the cone is diluted by (alpha/gamma)^2:
        L_avg = L * fov_dilution
    Sources filling the cone are undiluted (factor 1).
    """
    gamma = fov_gamma(t_sec)
    r = np.asarray(alpha_rad, dtype=float) / gamma
    return np.where(r >= 1.0, 1.0, r * r)


# --- Exposure limits ------------------------------------------------------------
def retinal_thermal_el(alpha_rad, t_sec):
    """Sec 4.3.5 EL: 5e4 / (alpha * t^0.25) [W m-2 sr-1] (10 us <= t <= 10 s)."""
    a = np.maximum(np.asarray(alpha_rad, dtype=float), ALPHA_MIN)
    t = np.maximum(np.asarray(t_sec, dtype=float), 1e-6)
    return THERMAL_EL_K / (a * np.power(t, 0.25))


def blue_light_el(t_sec):
    """Sec 4.3.3 radiance EL: 1e6/t for t<=1e4 s, 100 W m-2 sr-1 above."""
    t = np.asarray(t_sec, dtype=float)
    el = np.where(t <= 1.0e4, 1.0e6 / np.maximum(t, 1e-9), 100.0)
    return el


def blue_light_small_el(t_sec):
    """Sec 4.3.4 small-source irradiance EL: 100/t for t<=100 s (W m-2)."""
    return 100.0 / np.maximum(np.asarray(t_sec, dtype=float), 1e-9)


# --- Risk-group mapping ---------------------------------------------------------
def risk_group(value, limits):
    """Map a hazard value to 'Exempt'/'RG1'/'RG2'/'RG3' from an EL 3-tuple.

    Value <= Exempt -> Exempt; Exempt < v <= RG1 -> RG1; RG1 < v <= RG2 -> RG2;
    v > RG2 -> RG3 (High-Risk). For hazards whose Exempt boundary equals the
    RG1 boundary (retinal thermal) the first gap is empty by design.
    """
    v = np.asarray(value, dtype=float)
    ex, r1, r2 = limits
    out = np.full(np.shape(v), "Exempt", dtype="<U6")
    out[v > ex] = "RG1"
    out[v > r1] = "RG2"
    out[v > r2] = "RG3"
    return out


_RG_RANK = {"Exempt": 0, "RG1": 1, "RG2": 2, "RG3": 3}
_RG_KEYS = np.array(["Exempt", "RG1", "RG2", "RG3"])


def worst_risk_group(labels):
    """Element-wise worst risk across per-hazard group labels.

    `labels` is shaped (n,) for a single hazard or (n, k) for k hazards per
    sample (rows = glare minutes, last axis = hazards).
    """
    arr = np.asarray(labels)
    ranks = np.empty(arr.shape, dtype=int)
    for i, lb in enumerate(_RG_KEYS):
        ranks[arr == lb] = i
    worst = ranks.max(axis=arr.ndim - 1)
    return _RG_KEYS[worst]


# --- Per-glare-minute evaluation -------------------------------------------------
@stage_fn("evaluate_glare")
def evaluate_glare(ec_wcm2, er_wcm2, distance_m, width_m, height_m,
                   abs_cos_ref=1.0, weights=None):
    """Vectorised IEC 62471 evaluation of pre-computed glare minutes.

    Parameters mirror the SGHAT ``_analyze_single`` quantities for the same
    rows: E_c (W/cm2), E_r (W/cm2), observer distance (m), panel dims (m) and
    the reflected-ray/panel-normal |cos| fidelity factor.

    Returns a dict of arrays with the IEC columns used downstream:

      alpha_rad          : apparent source angular subtense (Sec 4.2.2)
      l_w_sr_m2          : reflected-beam radiance            [W m-2 sr-1]
      lr_w_sr_m2         : R-weighted (retinal thermal) radiance
      lb_w_sr_m2         : B-weighted (blue light) radiance
      eb_w_m2            : B-weighted irradiance (small source, Sec 4.3.4)
      lb_diluted_w_sr_m2 : L_B averaged over the FOV cone at T_AVERSION
      el_thermal_rg1     : 2.8e4/alpha [W m-2 sr-1] (Exempt/RG1, t=10 s)
      el_thermal_rg2     : 7.1e4/alpha [W m-2 sr-1] (RG2, t=0.25 s)
      tmax_thermal_s     : Eq 4.9 inversion
      tmax_blue_s        : Eq 4.6 inversion (radiance form)
      risk_thermal       : 'Exempt'/'RG1'/'RG2'/'RG3'
      risk_blue          : as above (radiance for alpha>11 mrad, E_B below)
      risk_eye           : worst of thermal + blue (FORMAL group)
      es_w_m2, euva_w_m2, eir_w_m2, eh_w_m2 : supplementary weighted
      er_lr_wcm2         : R-weighted retinal irradiance (k_R * E_r)
    """
    weights = weights or spectral_weights()
    kS, kUVA, kB, kR, kIR, kH = (weights[k] for k in ("kS", "kUVA", "kB", "kR", "kIR", "kH"))
    alpha = angular_subtense(width_m, height_m, distance_m, abs_cos_ref)
    L = source_radiance(ec_wcm2)
    L_R, L_B = kR * L, kB * L
    small = alpha < ALPHA_SMALL   # decision tree Sec 4.2.2
    E_B = kB * np.asarray(ec_wcm2, dtype=float) * 1e4
    # Blue: extended uses radiance with FOV dilution; small source uses E_B.
    lb_avg = L_B * fov_dilution(alpha, T_AVERSION)
    risk_blue = np.where(
        small,
        risk_group(E_B, (BLUE_EXEMPT_E, BLUE_RG1_E, BLUE_RG2_E)),
        risk_group(lb_avg, (BLUE_EXEMPT_RAD, BLUE_RG1_RAD, BLUE_RG2_RAD)),
    )
    risk_thermal = risk_group(L_R, (_THERMAL_EXEMPT_K / alpha,
                                    _THERMAL_EXEMPT_K / alpha,
                                    _THERMAL_RG2_K / alpha))
    return {
        "alpha_rad": alpha,
        "l_w_sr_m2": L,
        "lr_w_sr_m2": L_R,
        "lb_w_sr_m2": L_B,
        "eb_w_m2": E_B,
        "lb_diluted_w_sr_m2": lb_avg,
        "el_thermal_rg1_w_sr_m2": _THERMAL_EXEMPT_K / alpha,
        "el_thermal_rg2_w_sr_m2": _THERMAL_RG2_K / alpha,
        "tmax_thermal_s": _tmax_thermal(L_R, alpha),
        "tmax_blue_s": _tmax_blue(L_B),
        "risk_thermal": risk_thermal,
        "risk_blue": risk_blue,
        "risk_eye": worst_risk_group(np.column_stack([risk_thermal, risk_blue])),
        "es_w_m2": kS * np.asarray(ec_wcm2, dtype=float) * 1e4,
        "euva_w_m2": kUVA * np.asarray(ec_wcm2, dtype=float) * 1e4,
        "eir_w_m2": kIR * np.asarray(ec_wcm2, dtype=float) * 1e4,
        "eh_w_m2": kH * np.asarray(ec_wcm2, dtype=float) * 1e4,
        "er_lr_wcm2": kR * np.asarray(er_wcm2, dtype=float),
    }


def _tmax_thermal(lr_w_sr_m2, alpha_rad):
    """Safe t that just satisfies Eq 4.9: tmax = (5e4/(L_R*alpha))^4 s."""
    denom = np.maximum(np.asarray(lr_w_sr_m2, dtype=float) * np.asarray(alpha_rad, dtype=float), 1e-12)
    return np.power(THERMAL_EL_K / denom, 4.0)


def _tmax_blue(lb_w_sr_m2):
    """Safe t for blue light, Eq 4.6: tmax = 1e6 / L_B s (valid for t<=1e4)."""
    return 1.0e6 / np.maximum(np.asarray(lb_w_sr_m2, dtype=float), 1e-9)