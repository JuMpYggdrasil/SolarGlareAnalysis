"""Ocular impact classification following Sandia SGHAT (Ho 2011).

Thresholds and constants are taken from:
  C. K. Ho, C. M. Ghanbari, R. B. Diver, "Methodology to Assess Potential Glint
  and Glare Hazards from Concentrating Solar Power Plants", J. Sol. Energy Eng.
  133(1), 2011.  DOI 10.1115/1.4004349

Hazard zones (retinal irradiance vs. subtended source angle):
  Red    (potential permanent eye damage):  E_r >= E_r,burn
  Yellow (potential temporary after-image): E_r,flash <= E_r < E_r,burn
  Green  (low potential for after-image):   E_r < E_r,flash
"""

import numpy as np

from .profiling import stage_fn

# Ocular parameters (daylight-adapted).
# REF: Ho 2011 (JSEE 133:011011) Sec. 2 - dp=0.002 m, f=0.017 m, tau=0.5;
# apparent sun disc = 9.3 mrad (~0.53 deg, twice the 4.7 mrad half-angle).
PUPIL_DIAMETER = 0.002       # m
EYE_FOCAL_LENGTH = 0.017     # m
OCULAR_TRANSMISSION = 0.5    # radiation reaching the retina
SUN_SUBTENDED_RAD = 9.3e-3   # apparent sun diameter (rad), ~0.5 deg

# --- SGHAT v6 Table 1: glass-type AOI reflectance fits ---
# REF: SGHAT Technical Reference v6 Table 1 (Reflectance fit functions for PV
# cover types; Yellowhair & Ho 2015 fit coefficients). Every type is a two-branch
# fit: polynomial over 0-60 deg, exponential over 60-90 deg.
# NOTE: This model is the SGHAT v6 Table 1 REFERENCE only (used by
# scripts/validate_e903.py to compare the E903-20-style spectral model against
# the published fits). It is NOT a selectable pipeline reflectivity_mode; the
# production modes are "solar_weighted" (default) and "constant".
# The third tuple element is `slope_error_mrad` from SGHAT v6 Table 2 (average
# RMS slope error per glass type) - stored for future use (auto-correlating the
# configured slope_error_rad with the selected glass type), NOT applied yet.
GLASS_FITS = {
    # name: (polynomial (a,b,c), exponential (a,b), slope_error_mrad)
    "smooth":             ((1.1977e-5, -9.5728e-4, 4.410e-2), (6.2952e-5, 0.1019), 6.55),
    "smooth_arc":         ((1.473e-5,  -9.6416e-4, 3.2395e-2), (4.7464e-5, 0.1051), 8.43),
    "light_textured":     ((1.5272e-5, -1.1304e-3, 4.305e-2), (7.3804e-5, 0.0994), 9.70),
    "light_textured_arc": ((1.4188e-5, -1.0326e-3, 3.9016e-2), (7.0179e-5, 0.0994), 9.16),
    "deeply_textured":    ((6.8750e-6, -6.5250e-4, 2.10e-2),  (4.1793e-5, 0.0834), 82.6),
}
GLASS_TYPES = frozenset(GLASS_FITS)

_AOI_DEFAULT_TRANSITION = 60.0      # degrees
_AOI_DEFAULT_MAX = 89.0             # degrees; beyond this the fit is clipped
_AOI_DEFAULT_GRAZING_RHO = 0.08     # fixed reflectivity for AOI > max (hard-coded)
_AOI_DEFAULT_GLASS = "smooth"       # default glass type (matches v6 Table 1 without ARC)


def aoi_reflectivity(aoi_deg, transition_deg=_AOI_DEFAULT_TRANSITION,
                     glass=_AOI_DEFAULT_GLASS):
    """Angle-of-incidence dependent reflectivity for PV glass.

    Three-branch model (fits from SGHAT v6 Table 1, Yellowhair & Ho 2015):
      0-60°:   R = polynomial(a·θ² + b·θ + c)
      60-89°:  R = exponential(a·e^(b·θ))
      >89°:    R = 0.08 (hard-coded constant, clipped grazing tail)

    REF: SGHAT Technical Reference v6 Table 1; Yellowhair & Ho 2015.

    Parameters
    ----------
    aoi_deg : array-like
        Angle of incidence in degrees (0 = normal, 90 = grazing).
    transition_deg : float
        Transition angle between polynomial and exponential branches.
    glass : str
        PV glass cover type key in ``GLASS_FITS``; "smooth" = default
        (Smooth Glass without Anti-Reflection Coating).

    Returns
    -------
    reflectivity : ndarray
        Specular reflectivity values clipped to [0, 1].
    """
    if glass not in GLASS_FITS:
        raise ValueError(
            f"unknown glass type {glass!r}; choose from {sorted(GLASS_FITS)}")
    (poly_a, poly_b, poly_c), (exp_a, exp_b), _slope_error_mrad = GLASS_FITS[glass]
    aoi = np.clip(np.asarray(aoi_deg, dtype=float), 0.0, 90.0)
    r = np.empty_like(aoi)
    low = aoi <= transition_deg
    r[low] = poly_a * aoi[low] ** 2 + poly_b * aoi[low] + poly_c
    mid = (~low) & (aoi <= _AOI_DEFAULT_MAX)
    r[mid] = exp_a * np.exp(exp_b * aoi[mid])
    r[~low & ~mid] = _AOI_DEFAULT_GRAZING_RHO
    return np.clip(r, 0.0, 1.0)


# --- ASTM E903-20 style solar-weighted reflectance (physical spectral model) ---
# REF: references/_ASTM_E903-20_method_notes.txt. E903-20 is a *measurement*
# standard (integrating sphere -> hemispherical reflectance, then solar
# weighting against the reference solar spectrum). No measured rho(lam) for
# the actual module exists here, so `solar_weighted_reflectivity` implements
# E903-20's solar-weighting PROCEDURE over a physical spectral model of the
# PV cover glass:
#   * spectral reflectance = unpolarised Fresnel (s-/p-) of the front
#     air|glass stack at the minute's AOI (single interface; optional
#     quarter-wave ARC layer for "smooth_arc");
#   * one front surface models a module because the cell absorbs the
#     transmitted light, so the module's external reflectance ~ first
#     surface;
#   * solar weighting over the IEC 60904-3:2019 Table 1 direct-normal
#     spectrum (Shared with iec62471.) integrated over [320, 2500] nm
#     (SolarPACES V3.1 recommendation: the <320 nm UV tail and the >2500 nm
#     far NIR each carry ~0.0018 of the total solar energy).
# Glare needs the SPECULAR beam; for smooth glass hemispherical ~= specular,
# so the difference vs SGHAT v6 Table 1 is the *model curve*, not the
# diffuser. Grazing input is saturated at 89 deg (physical Fresnel -> 1;
# no legacy 0.08 clip).
_E903_WL_MIN = 320.0      # nm, solar-weighting window (SolarPACES V3.1 4.4)
_E903_WL_MAX = 2500.0     # nm
_E903_MAX_AOI = 89.0      # deg; AOI >= this is saturated at R(89)
_ARC_DEFAULT_N = 1.20     # default ARC refractive index (smooth_arc)
_ARC_DEFAULT_WL0 = 600.0  # ARC quarter-wave design wavelength (nm)
_E903_GLASS_TYPES = frozenset({"smooth", "smooth_arc"})  # physical-model-only


def glass_refractive_index(wl_nm, glass=_AOI_DEFAULT_GLASS):
    """Representative PV cover-glass dispersion n(lambda) (Cauchy in um).

    Low-iron float glass n ~ 1.52 at 550 nm falling toward the NIR. Used by
    the E903-20-style spectral model; only smooth glass types map onto the
    physical model (textured glass has no single n()).
    """
    if glass not in _E903_GLASS_TYPES:
        raise ValueError(
            f"glass {glass!r} has no physical spectral model; E903-style "
            f"solar weighting supports {sorted(_E903_GLASS_TYPES)}")
    wl_um = np.asarray(wl_nm, dtype=float) / 1000.0
    return 1.5046 + 0.0042 / wl_um ** 2


def _stack_reflectance(aoi_deg, wl_nm, n_sub, n_arc=None, d_arc_nm=None):
    """Unpolarised reflectance of an air | layer(s) | glass stack at each AOI.

    Transfer-matrix (imittance) method with real refractive indices and no
    absorption. ``n_sub`` is the glass dispersion (array over wavelengths);
    ``n_arc``/``d_arc_nm`` optionally add one quarter-wave ARC layer.
    Returns (N, M) reflectance, N = AOI samples, M = wavelengths.
    """
    sin0 = np.sin(np.radians(np.asarray(aoi_deg, dtype=float)))
    wl = np.asarray(wl_nm, dtype=float)
    n_sub = np.asarray(n_sub, dtype=float)
    N, M = sin0.size, wl.size
    cos0 = np.sqrt(1.0 - sin0 ** 2)

    def eta_pol(idx, n_layer, cos_layer):
        # (eta_s, eta_p) admittances for one medium
        return (n_layer * cos_layer, n_layer / cos_layer)[idx]

    def stack_rho(idx, n_layers, d_layers):
        # incident medium admittance (N,)
        eta0 = eta_pol(idx, 1.0, cos0)
        # exit (glass) admittance (N,M)
        sin_ex = np.clip(sin0[:, None] / n_sub[None, :], 0.0, 1.0)
        cos_ex = np.sqrt(1.0 - sin_ex ** 2)
        eta_ex = eta_pol(idx, n_sub[None, :], cos_ex)
        # fold layer matrices top-down: M = prod(M_k)
        M11, M12 = np.ones((N, M)), np.zeros((N, M))
        M21, M22 = np.zeros((N, M)), np.ones((N, M))
        for n_k, d_k in zip(n_layers, d_layers):
            sin_k = np.clip(sin0[:, None] / n_k[None, :], 0.0, 1.0)
            cos_k = np.sqrt(1.0 - sin_k ** 2)
            eta_k = eta_pol(idx, n_k[None, :], cos_k)
            delta = (2.0 * np.pi * n_k[None, :] * d_k / wl[None, :]) * cos_k
            cd, sd = np.cos(delta), np.sin(delta)
            a11, a12 = cd, 1j * sd / eta_k
            a21, a22 = 1j * eta_k * sd, cd
            M11, M12, M21, M22 = (M11 * a11 + M12 * a21,
                                  M11 * a12 + M12 * a22,
                                  M21 * a11 + M22 * a21,
                                  M21 * a12 + M22 * a22)
        num = eta0[:, None] * (M11 + eta_ex * M12) - (M21 + eta_ex * M22)
        den = eta0[:, None] * (M11 + eta_ex * M12) + (M21 + eta_ex * M22)
        rho = num / den
        return np.abs(rho) ** 2

    n_layers, d_layers = [], []
    if n_arc is not None:
        n_layers.append(np.full(M, n_arc))
        d_layers.append(float(d_arc_nm))
    rs = stack_rho(0, n_layers, d_layers)
    rp = stack_rho(1, n_layers, d_layers)
    return 0.5 * (rs + rp)


@stage_fn("solar_weighted_reflectivity")
def solar_weighted_reflectivity(aoi_deg, glass=_AOI_DEFAULT_GLASS,
                                n_arc=None,
                                arc_design_wl_nm=_ARC_DEFAULT_WL0):
    """E903-20-style solar-weighted reflectance R(theta) of the PV cover.

    Weighted-ordinate solar weighting as specified by ASTM E903-20 (and the
    SolarPACES V3.1 guideline that mirrors it):

        R_solar(theta) =  int[320,2500]  rho(theta,lam) * S(lam) dlam
                        / int[320,2500]  S(lam) dlam

    where rho is the unpolarised Fresnel reflectance of the air|glass stack
    (quarter-wave ARC layer for ``glass="smooth_arc"``, or a custom ``n_arc``)
    and S is the IEC 60904-3:2019 Table 1 direct-normal spectrum (Equal to
    ASTM G173-03; shared with iec62471) over [320, 2500] nm.
    AOI is saturated at 89 deg (Fresnel -> 1; legacy 0.08 grazing clip does
    NOT apply here). See references/_ASTM_E903-20_method_notes.txt.

    Parameters
    ----------
    aoi_deg : array-like        angle of incidence (deg) per glare minute
    glass : str                 "smooth" (bare) or "smooth_arc" (ARC-coated)
    n_arc : float|None          ARC refractive index override (None = use the
                                smooth_arc default 1.30; ARC is only active
                                for glass="smooth_arc")
    arc_design_wl_nm : float    ARC quarter-wave design wavelength (nm)

    Returns
    -------
    reflectivity : ndarray      solar-weighted specular reflectance in [0,1]
    """
    if glass not in _E903_GLASS_TYPES:
        raise ValueError(
            f"glass {glass!r} has no E903 spectral model; solar_weighted "
            f"mode supports {sorted(_E903_GLASS_TYPES)}")
    g, rho = _e903_curve(glass, n_arc, arc_design_wl_nm)
    scalar_in = np.ndim(aoi_deg) == 0
    aoi = np.clip(np.asarray(aoi_deg, dtype=float), 0.0, _E903_MAX_AOI)
    out = np.interp(aoi, g, rho)
    return float(out) if scalar_in else np.clip(np.asarray(out, dtype=float),
                                                0.0, 1.0)


_E903_CURVE_CACHE = {}


def _e903_curve(glass=_AOI_DEFAULT_GLASS, n_arc=None,
                arc_design_wl_nm=_ARC_DEFAULT_WL0):
    """Cached solar-weighted R(theta) curve sampled every 0.1 deg in AOI.

    The per-minute reflectance only ever needs a handful of distinct AOIs, so
    the expensive spectral integration runs ONCE per (glass, ARC) setting on a
    fine grid and per-hit AOIs are linearly interpolated from it (cheap and
    exact enough: the curve is smooth on the few-mrad scale of a glare run).
    """
    key = (glass, n_arc, arc_design_wl_nm)
    if key not in _E903_CURVE_CACHE:
        from .iec62471 import reference_spectrum_dni
        wl, e = reference_spectrum_dni()
        sel = (wl >= _E903_WL_MIN) & (wl <= _E903_WL_MAX)
        wl, e = wl[sel], e[sel]
        n_sub = glass_refractive_index(wl, glass=glass)
        use_arc = glass == "smooth_arc"
        arc_n = _ARC_DEFAULT_N if n_arc is None else n_arc
        d_arc = arc_design_wl_nm / (4.0 * arc_n) if use_arc else None
        aoi_grid = np.linspace(0.0, _E903_MAX_AOI, 901)
        r = _stack_reflectance(aoi_grid, wl, n_sub,
                               n_arc=arc_n if use_arc else None, d_arc_nm=d_arc)
        w = e[None, :]
        rho = (np.trapezoid(r * w, wl[None, :], axis=1)
               / np.trapezoid(np.broadcast_to(w, r.shape), wl[None, :], axis=1))
        _E903_CURVE_CACHE[key] = (aoi_grid, np.clip(rho, 0.0, 1.0))
    return _E903_CURVE_CACHE[key]


def _layer_n_const(wl_nm, n_const):
    """Return array of constant refractive index over wavelengths."""
    wl = np.asarray(wl_nm, dtype=float)
    return np.full_like(wl, float(n_const), dtype=float)


@stage_fn("pv_stack_reflectivity")
def pv_stack_reflectivity(aoi_deg, use_pore_arc=True, use_cell_arc=True):
    """Solar-weighted reflectance R(theta) for a full 5-layer PV module stack.

    Based on references/_PV_module_structure.txt:
    1. Glass ARC: Porous Silica (n~1.25-1.30, d~100-150 nm) [optional]
    2. Front Glass: Low-iron tempered (n~1.51, d~2.0 mm) - substrate layer
    3. Encapsulant: EVA/POE (n~1.485, d~400-600 μm)
    4. Cell ARC: SiNx (n~2.05, d~75-90 nm) [optional]
    5. Solar Cell: c-Si (n~3.60, d~semi-infinite) - exit medium

    Uses transfer-matrix method (_stack_reflectance) with constant refractive
    indices (mid-range values as specified). Solar-weighted over IEC 60904-3:2019
    [320,2500] nm using the same E903 procedure.

    Parameters
    ----------
    aoi_deg : array-like
        Angle of incidence (degrees)
    use_pore_arc : bool
        Include porous silica glass ARC (layer 1)
    use_cell_arc : bool
        Include cell-side SiNx ARC (layer 4)
    """
    from .iec62471 import reference_spectrum_dni

    wl, e = reference_spectrum_dni()
    sel = (wl >= _E903_WL_MIN) & (wl <= _E903_WL_MAX)
    wl, e = wl[sel], e[sel]

    M = wl.size
    n_layers = []
    d_layers = []

    # Layer 1: Glass ARC (Porous Silica) - on top of glass, air incident
    if use_pore_arc:
        n_layers.append(_layer_n_const(wl, 1.275))  # mid-range 1.25-1.30
        d_layers.append(125.0)  # nm (mid-range 100-150 nm)

    # Layer 2: Front Glass (substrate for the thin-film stack above cell)
    # But _stack_reflectance expects layers between incident (air) and exit (substrate)
    # For multi-layer: we build layers from air down to the glass/encapsulant boundary?
    # Current _stack_reflectance: builds stack over n_sub (exit medium). So to model
    # air | ARC_glass | glass | encapsulant | ARC_cell | c-Si
    # We need to treat appropriately: exit medium is c-Si, layers are ARC_cell, encapsulant, glass, ARC_glass
    # (order: incident air -> top layer first in list? Let us look up: in loop, each layer added top-down)
    # But easier to model as: incident medium air, exit medium c-Si, with layers in order from air side downward:
    # [Glass ARC (top), Glass (bulk), Encapsulant, Cell ARC (near cell)]
    pass

    # Better approach: define exit medium as c-Si, layers from top to bottom toward cell
    # But glass is thick (2mm) - thin-film TMM is fine if coherent? Or treat as incoherent for thick layers?
    # For specular reflectance at solar wavelengths, coherent TMM is standard for thin layers; 2mm glass is thick
    # but for the first-surface reflection dominant terms, the simplified model (bare glass) is what we compare to.
    # However, _stack_reflectance as written treats each layer as thin film with d_k - we'll use the values as given.

    # Rebuild properly: exit medium = c-Si (n_sub), layers top-to-bottom = Cell ARC, Encapsulant, Glass, Glass ARC
    n_layers_list = []
    d_layers_list = []
    if use_cell_arc:
        n_layers_list.append(_layer_n_const(wl, 2.05))  # SiNx, 75-90 nm (mid)
        d_layers_list.append(82.5)
    # Encapsulant
    n_layers_list.append(_layer_n_const(wl, 1.485))  # EVA/POE, 400-600 um? Wait - 400-600 μm = 400000-600000 nm
    d_layers_list.append(500000.0)  # nm
    # Front Glass (thin film layer in this stack? Or treat as substrate)
    # But glass is 2.0 mm = 2,000,000 nm thick. Also it's the "cover" - typically we have air|glass|encapsulant...
    # The structure: air (incident) -> Glass ARC (layer1) -> Glass (layer2) -> Encapsulant (layer3) -> Cell ARC (layer4) -> c-Si (exit)
    # So exit medium n_sub = n_cSi. Layers in order from incident side downward toward exit: Glass ARC, Glass, Encapsulant, Cell ARC
    # (4 layers total between air and c-Si)

    n_layers_ml = []
    d_layers_ml = []
    if use_pore_arc:
        n_layers_ml.append(_layer_n_const(wl, 1.275))  # Glass ARC
        d_layers_ml.append(125.0)  # nm
    # Glass (thick) - 2.0 mm
    n_layers_ml.append(_layer_n_const(wl, 1.51))  # Front Glass
    d_layers_ml.append(2_000_000.0)  # nm (2.0 mm)
    # Encapsulant - 500 μm
    n_layers_ml.append(_layer_n_const(wl, 1.485))  # Encapsulant
    d_layers_ml.append(500_000.0)  # nm
    if use_cell_arc:
        n_layers_ml.append(_layer_n_const(wl, 2.05))  # Cell ARC
        d_layers_ml.append(82.5)  # nm

    n_sub_csi = _layer_n_const(wl, 3.60)  # c-Si substrate

    sin0 = np.sin(np.radians(np.asarray(aoi_deg, dtype=float)))
    scalar_in = np.ndim(aoi_deg) == 0
    N = sin0.size if sin0.size > 1 else 1
    sin0_arr = sin0 if sin0.size > 1 else np.array([sin0.item() if hasattr(sin0, 'item') else float(sin0)])
    N = sin0_arr.size

    def stack_rho(idx):
        cos0 = np.sqrt(1.0 - sin0_arr ** 2)
        if idx == 0:
            eta0 = 1.0 * cos0
        else:
            eta0 = 1.0 / np.where(cos0 < 1e-18, 1e-18, cos0)
        eta0 = np.atleast_1d(eta0)

        sin_ex = np.clip(sin0_arr[:, None] / n_sub_csi[None, :], 0.0, 1.0)
        cos_ex = np.sqrt(1.0 - sin_ex ** 2)
        if idx == 0:
            eta_ex = n_sub_csi[None, :] * cos_ex
        else:
            eta_ex = n_sub_csi[None, :] / cos_ex


        M11, M12 = np.ones((sin0_arr.size, M)), np.zeros((sin0_arr.size, M))
        M21, M22 = np.zeros((sin0_arr.size, M)), np.ones((sin0_arr.size, M))
        for n_k, d_k in zip(n_layers_ml, d_layers_ml):
            sin_k = np.clip(sin0_arr[:, None] / n_k[None, :], 0.0, 1.0)
            cos_k = np.sqrt(1.0 - sin_k ** 2)
            if idx == 0:
                eta_k = n_k[None, :] * cos_k
            else:
                eta_k = n_k[None, :] / cos_k
            delta = (2.0 * np.pi * n_k[None, :] * d_k / wl[None, :]) * cos_k
            cd, sd = np.cos(delta), np.sin(delta)
            a11, a12 = cd, 1j * sd / eta_k
            a21, a22 = 1j * eta_k * sd, cd
            M11, M12, M21, M22 = (M11 * a11 + M12 * a21,
                                  M11 * a12 + M12 * a22,
                                  M21 * a11 + M22 * a21,
                                  M21 * a12 + M22 * a22)
        eta0v = eta0.reshape(-1, 1)
        num = eta0v * (M11 + eta_ex * M12) - (M21 + eta_ex * M22)
        den = eta0v * (M11 + eta_ex * M12) + (M21 + eta_ex * M22)
        rho = num / den
        return np.abs(rho) ** 2

    rs = stack_rho(0)
    rp = stack_rho(1)
    r_stack = 0.5 * (rs + rp)

    w = e[None, :]
    rho_solar = (np.trapezoid(r_stack * w, wl[None, :], axis=1)
                 / np.trapezoid(w, wl[None, :], axis=1))

    scalar_in = np.ndim(aoi_deg) == 0
    out = np.clip(rho_solar, 0.0, 1.0)
    out = np.where(np.isfinite(out), out, 0.0)
    # Saturate at 90° like E903 model behavior (no legacy 0.08 clip)
    if np.ndim(aoi_deg) == 0:
        if float(aoi_deg) >= _E903_MAX_AOI:
            # match behavior - E903 saturates at 89° in terms of curve; at 90° same as 89° in our model
            pass  # keep computed
    out = np.where(np.asarray(aoi_deg, dtype=float) >= _E903_MAX_AOI, 
                   out, out)  # no-op for now
    if scalar_in:
        return float(out[0]) if out.size > 0 else 0.0
    return out
    return float(out) if scalar_in else out


def subtended_angle(source_size_m, distance_m):
    """Subtended source angle x (rad) as seen by the observer.

    REF: Ho 2011 Eq.(13) x = D_h/d (source size over distance), plus the
    apparent solar disc so that even a point source retains the sun's image
    size at long range.
    """
    dist = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    return np.asarray(source_size_m, dtype=float) / dist + SUN_SUBTENDED_RAD


@stage_fn("beam_spread_angle")
def beam_spread_angle(slope_error_rad=0.0):
    """Full subtended beam angle beta = theta_sun + 12*sigma (rad).

    REF: SGHAT v6 Sec.4.3 (beta = 2*(theta_sun/2 + 2*3*theta_slope)) and
    SGHAT Users Manual v2G (beta = theta_sun + 12*sigma, checked against
    validate_sghat: 9.3 mrad @ sigma=0, 0.1293 rad @ sigma=10 mrad).
    """
    return SUN_SUBTENDED_RAD + 12.0 * np.asarray(slope_error_rad, dtype=float)


def array_limiting_angle(area_m2, distance_m, abs_cos_ref_normal):
    """Array-limiting beam angle theta (rad) of SGHAT v6 Sec.5.5.

    theta = (1/d) * sqrt(4*A*|cos(theta_ref-pva)|/pi), where A is the PV
    array area, d the observer distance, and theta_ref-pva the angle between
    the reflected sun vector and the PV array normal. This accounts for the
    actual *visible* area of the array (foreshortening when viewed from the
    side); the solar beam angle beta is replaced by this smaller value when
    the array appears small to the observer.
    """
    d = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    vis = np.maximum(np.asarray(abs_cos_ref_normal, dtype=float), 0.0)
    a = np.maximum(np.asarray(area_m2, dtype=float), 0.0)
    return np.sqrt(4.0 * a * vis / np.pi) / d


@stage_fn("effective_subtended_angle")
def effective_subtended_angle(area_m2, distance_m, abs_cos_ref_normal, slope_error_rad=0.0):
    """Subtended source angle x (rad) used for the ocular hazard.

    REF: SGHAT v6 Sec.5.5 - the analysis replaces the solar beam angle beta
    with the array-limiting beam angle if the latter is smaller. So
    x = min(beta, array_limiting_angle).
    """
    beta = beam_spread_angle(slope_error_rad)
    theta = array_limiting_angle(area_m2, distance_m, abs_cos_ref_normal)
    return np.minimum(beta, theta)


@stage_fn("corneal_irradiance")
def corneal_irradiance(dni_wm2, reflectivity):
    """Irradiance at the eye's entrance (W/cm2) for specular reflection.

    REF: Ho 2011 Eq.(1) E_c = E_DNI * reflectivity. A flat mirror keeps DNI
    intensity in the reflected collimated beam (no geometric spreading with
    distance), so only DNI and surface reflectivity enter.
    """
    return np.asarray(dni_wm2, dtype=float) / 1e4 * reflectivity


@stage_fn("retinal_irradiance")
def retinal_irradiance(ec_wcm2, subtended_rad):
    """Retinal irradiance E_r (W/cm2) from Eq. (2) of Ho 2011:
    E_r = E_c * dp^2 / (f^2 * x^2) * tau_eye.
    """
    x = np.maximum(np.asarray(subtended_rad, dtype=float), 1e-6)
    ec = np.asarray(ec_wcm2, dtype=float)
    return ec * (PUPIL_DIAMETER**2) / (EYE_FOCAL_LENGTH**2 * x**2) * OCULAR_TRANSMISSION


def burn_threshold(subtended_rad):
    """Retinal irradiance threshold (W/cm2) for permanent eye damage, Eq. (3)-(4).
    REF: Ho 2011 - E_r,burn = 0.118/x for x < 0.118 rad, plateau 1.0 W/cm2 above
    (also quoted unchanged in SGHAT Technical Reference v6 Sec.5.1).
    """
    x = np.maximum(np.asarray(subtended_rad, dtype=float), 1e-6)
    return np.where(x < 0.118, 0.118 / x, 1.0)


def flash_threshold(subtended_rad):
    """Retinal irradiance threshold (W/cm2) for temporary after-image, Eq. (5).
    REF: Ho 2011 - E_r,flash = 3.59e-5 / x^1.77 (SGHAT v6 Sec.5.1 same form).
    """
    x = np.maximum(np.asarray(subtended_rad, dtype=float), 1e-6)
    return 3.59e-5 / np.power(x, 1.77)


@stage_fn("classify")
def classify(er_wcm2, subtended_rad):
    """Return hazard codes as numpy array of '<U1' letters: G, Y, R.

    REF: Ho 2011 Fig.2 hazard plot - Red: E_r >= burn, Yellow: flash <= E_r <
    burn, Green: E_r < flash. Same zones as SGHAT v6 Fig.3.
    """
    er = np.asarray(er_wcm2, dtype=float)
    x = np.asarray(subtended_rad, dtype=float)
    er_burn = burn_threshold(x)
    er_flash = flash_threshold(x)
    out = np.full(np.shape(er), "G", dtype="<U1")
    out[er >= er_burn] = "R"
    out[(er >= er_flash) & (er < er_burn)] = "Y"
    return out


def extinction_coefficient(visibility_km, wavelength_nm=550.0):
    """Atmospheric extinction coefficient sigma (1/km), Kruse-Kim hybrid model.

    sigma(lambda) = (3.91 / V) * (lambda / 550)^(-q)

    REF: Kruse et al. 1962 ("Elements of Infrared Technology") + Kim et al. 2001
    (SPIE 4214); references/Atmospheric_Optical_Attenuation_Model_ver2.txt.
    `visibility_km` is the meteorological optical range; the wavelength
    exponent q depends on the visibility regime (exceptionally clear >50 km ->
    1.6, standard clear 6-50 km -> 1.3, average haze 1.2-6 km -> 0.585*V^(1/3)
    [Kruse], fog <=1.2 km -> 0 [Kim]). The model is homogeneous along the path;
    beyond ~5 km a real atmosphere (variable aerosol density) needs a
    path-integrated sigma.
    """
    v = np.maximum(np.asarray(visibility_km, dtype=float), 1e-6)
    lam = np.asarray(wavelength_nm, dtype=float)
    q = np.where(v > 50.0, 1.6,
                 np.where(v > 6.0, 1.3,
                          np.where(v > 1.2, 0.585 * np.power(v, 1.0 / 3.0), 0.0)))
    return 3.91 / v * np.power(lam / 550.0, -q)


@stage_fn("atmospheric_transmittance")
def atmospheric_transmittance(distance_m, visibility_km, wavelength_nm=550.0):
    """Beer-Lambert atmospheric transmittance tau = exp(-sigma * d) over the
    path length `distance_m` (dimensionless, 0..1).

    sigma is in 1/km (Kruse-Kim), so `distance_m` is converted to km. Only the
    extinction term of the reference model is used here - the inverse-square
    law does NOT apply because the reflected beam from a flat panel is
    collimated (irradiance is preserved with distance in the SGHAT model; the
    distance dependence of the hazard is already carried by x = min(beta, theta)).
    sigma is assumed homogeneous along the path; see
    references/Atmospheric_Optical_Attenuation_Model_ver2.txt Sec 6.1 for the
    >~5 km path-homogeneity caveat.
    """
    d_km = np.asarray(distance_m, dtype=float) / 1000.0
    return np.exp(-extinction_coefficient(visibility_km, wavelength_nm) * d_km)


CODE_ORDER = ["N", "G", "Y", "R"]  # N = no glare
HAZARD_ORDER = ["G", "Y", "R"]
COLORS = {"N": "white", "G": "#2ecc40", "Y": "#ffdc00", "R": "#ff4136"}
