"""Sanity checks for the glare engine (no test framework configured yet).

Run from repo root inside the venv:
    .venv\\Scripts\\python.exe scripts\\sanity_checks.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from solar_glare.geometry import (sun_vector, surface_normal, reflect, angle_between,
                                  panel_axes, panel_corners_enu, glare_polygon_hit)
from solar_glare.hazard import (
    retinal_irradiance, corneal_irradiance, subtended_angle,
    flash_threshold, burn_threshold, classify,
    array_limiting_angle, effective_subtended_angle,
    aoi_reflectivity,
    solar_weighted_reflectivity,
    glass_refractive_index,
    extinction_coefficient,
    atmospheric_transmittance,
    _E903_GLASS_TYPES,
)


def approx(a, b, tol=1e-9, msg=""):
    assert abs(a - b) <= tol, f"{msg}: {a} != {b}"


def test_mirror_law():
    # r = 2(s.n)n - s  <=>  (s + r) parallel to n  and  |r| = |s|
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.normal(size=3); s /= np.linalg.norm(s)
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        if s @ n < 0:
            n = -n
        r = reflect(s, n)
        cross = np.cross(s + r, n)
        approx(np.linalg.norm(cross), 0.0, 1e-9, "s + r not parallel to n")
        approx(np.linalg.norm(r), 1.0, 1e-12, "r not unit")
    print("OK mirror law (20 random cases)")


def test_normal_incidence():
    # Sun along the surface normal -> ray reflects straight back at the sun
    n = surface_normal(45.0, 180.0)  # facing south, 45 deg
    s = sun_vector(180.0, 45.0)       # sun at due south, 45 deg elevation
    r = reflect(s, n)
    approx(np.dot(r, s), 1.0, 1e-9, "normal incidence should return along path")
    print("OK normal incidence")


def test_known_panel_geometry():
    # Tilt 20, facing south; check normal components and unit length
    n = surface_normal(20.0, 180.0)
    approx(np.linalg.norm(n), 1.0, 1e-12)
    approx(n[0], 0.0, 1e-12, "east component")
    approx(n[1], -math.sin(math.radians(20)), 1e-12, "north component")
    approx(n[2], math.cos(math.radians(20)), 1e-12, "up component")
    print("OK panel normal (tilt 20, facing south)")


def test_retinal_irradiance_formula():
    # Eq. (2): same power onto a smaller retinal image -> higher E_r as x shrinks
    ec = corneal_irradiance(1000.0, 0.04)  # 0.004 W/cm2
    x1 = subtended_angle(100.0, 2000.0)
    x2 = subtended_angle(100.0, 4000.0)
    assert x2 < x1, "subtended angle must shrink with distance"
    er1 = retinal_irradiance(ec, x1)
    er2 = retinal_irradiance(ec, x2)
    assert er1 < er2, "retinal irradiance should grow as subtended angle shrinks"
    assert np.isfinite(er1) and np.isfinite(er2)
    print(f"OK retinal irradiance: Ec={ec:.4f} W/cm2, x1={x1:.4f} Er1={er1:.2e}, "
          f"x2={x2:.4f} Er2={er2:.2e}")


def test_threshold_shape():
    x = np.geomspace(1e-3, 1.0, 100)
    fb, bb = flash_threshold(x), burn_threshold(x)
    assert np.all(fb < bb), "after-image threshold must lie below burn threshold"
    # burn threshold: constant 1 W/cm2 above x >= 0.118 rad
    big = burn_threshold(0.2)
    approx(float(big), 1.0, 1e-12, "burn threshold above 0.118 rad")
    print("OK thresholds (flash < burn, plateau at 1 W/cm2)")


def test_classify_zones():
    x = np.array([0.01, 0.01, 0.01])  # flash thr ~0.125, burn thr = 11.8 W/cm2
    er = np.array([1e-8, 1.0, 50.0])  # green, yellow, red
    hz = classify(er, x)
    assert hz[0] == "G" and hz[1] == "Y" and hz[2] == "R", hz
    print("OK classify zones (G/Y/R)")


def test_sun_position_spa_eot():
    # NREL SPA equation of time for 21 Jan must be ~ -11.2 min. The simplified
    # Duffie & Beckman formula as printed in SGHAT v6 Sec.4.1 yields +16.4 min
    # (~27 min / ~6.8 deg hour-angle error) - the whole reason we use pvlib SPA.
    import pandas as pd
    import pvlib
    lat, lon = 12.6709119, 101.1157993
    t = pd.Timestamp("2025-01-21 08:00", tz="Asia/Bangkok")
    pos = pvlib.solarposition.get_solarposition(t, lat, lon, method="nrel_numpy")
    eot = float(pos["equation_of_time"])
    assert -13.0 < eot < -9.5, f"EOT {eot:.2f} min out of expected ~[-13,-9.5] (paper gives +16.4)"
    print(f"OK NREL SPA equation-of-time 21 Jan = {eot:.2f} min (rejects paper's misprinted +16.4)")
    assert 5.0 < float(pos["apparent_elevation"]) < 20.0, "8am apparent elevation sanity"
    print(f"OK NREL SPA apparent elevation 21 Jan 08:00 = {float(pos['apparent_elevation']):.2f} deg")


def test_heatmap_matrix_shape():
    from solar_glare.viz import _code_matrix
    import pandas as pd
    tz = "Asia/Bangkok"
    times = pd.date_range("2025-06-15", periods=3, freq="h", tz=tz)
    df = pd.DataFrame({"time": times, "hazard": ["Y", "N", "G"]})
    m = _code_matrix(df)
    # 30-min resolution: 365 days x 48 half-hour cells
    assert m.shape == (365, 48), m.shape
    assert m[165, 0] == "Y" and m[165, 2] == "N" and m[165, 4] == "G"
    print("OK heatmap matrix shape")


def test_array_limiting_angle():
    # SGHAT v6 Sec.5.5: theta = (1/d)*sqrt(4*A*|cos|/pi).
    # PV-B demo (A=2000 m2) face-on at 3260 m:
    t = array_limiting_angle(2000.0, 3260.0, 1.0)
    exp = math.sqrt(4.0 * 2000.0 / math.pi) / 3260.0
    assert abs(t - exp) < 1e-12, f"array-limiting {t} != {exp}"
    # Foreshortening: edge-on view shrinks the visible angle.
    t_side = array_limiting_angle(2000.0, 3260.0, 0.05)
    assert t_side < t * 0.3, f"edge-on angle {t_side} not << face-on {t}"
    # Effective x = min(beta, array-limiting).
    from solar_glare.hazard import beam_spread_angle
    x_large = effective_subtended_angle(2000.0, 3260.0, 1.0, 0.0)
    assert abs(x_large - beam_spread_angle(0.0)) < 1e-12, "large array should be beam-limited"
    x_small = effective_subtended_angle(50.0, 5000.0, 0.2, 0.0)
    assert x_small < beam_spread_angle(0.0), "small/off-axis array should be array-limited"
    print(f"OK array-limiting angle: theta face-on={t:.4f} rad, "
          f"edge-on={t_side:.4f} rad, x_large={x_large:.4e}, x_small={x_small:.4e}")


def test_front_face_sun_disc_guard():
    # A near-horizontal panel (tilt ~0) with the sun grazing its plane must NOT
    # glare: the front/back of the panel is ambiguous and part of the reflected
    # cone would come from behind the array. A clearly-lit sun must still glare.
    import pandas as pd
    from solar_glare.analysis import analyze_pair

    class FakeArray:
        def __init__(self, normal):
            self.normal = normal
            self.char_size = 10.0
            self.specular_reflectivity = 0.05
            self.center_enu = np.array([0.0, 0.0, 0.0])

    class FakeDni:
        def __init__(self, n=1):
            self._series = pd.Series([1000.0] * n)
        @property
        def iloc(self):
            return self._series.iloc
        def to_numpy(self):
            return self._series.to_numpy()

    class FakeReceptor:
        def __init__(self, enu):
            self.enu = np.array(enu, dtype=float)

    def fake_pos(az, el):
        return pd.DataFrame(
            {"azimuth": np.array([az]), "apparent_elevation": np.array([el])},
            index=np.array([0]))

    arr = FakeArray(np.array([0.0, 0.0, 1.0]))  # horizontal panel, front = up

    # Sun 0.1 deg above the panel plane (its disc straddles the plane) -> reject.
    s_low = sun_vector(90.0, 0.1)
    r_low = reflect(s_low, arr.normal)
    codes_low, det_low = analyze_pair(
        arr, FakeReceptor(r_low * 2000.0), fake_pos(90.0, 0.1), FakeDni(1))
    assert codes_low[0] == "N" and len(det_low) == 0, f"grazing sun: {codes_low[0]}"

    # Sun 5 deg above the panel plane -> clearly front-lit -> hit.
    s_hi = sun_vector(90.0, 5.0)
    r_hi = reflect(s_hi, arr.normal)
    codes_hi, det_hi = analyze_pair(
        arr, FakeReceptor(r_hi * 2000.0), fake_pos(90.0, 5.0), FakeDni(1))
    assert codes_hi[0] in ("G", "Y", "R") and len(det_hi) == 1, f"clear sun: {codes_hi[0]}"
    print("OK front-face sun-disc guard (grazing sun rejected, clear sun accepted)")


def test_aoi_reflectivity():
    # SGHAT v6 Table 1 smooth-glass fit: ~4.4% at normal incidence, dips to a
    # Brewster-style minimum mid-range, then rises steeply on the exponential
    # branch past ~60 degrees.
    r0 = float(aoi_reflectivity(0.0))
    assert abs(r0 - 0.0441) < 1e-4, f"R(0)={r0} != ~0.0441"
    r_min = float(aoi_reflectivity(40.0))
    assert r_min < r0, "reflectance must dip below normal-incidence (Brewster minimum)"
    r60 = float(aoi_reflectivity(60.0))
    assert 0.02 < r60 < 0.04, f"R(60)={r60}"
    r70, r80, r85, r89, r90 = (float(aoi_reflectivity(a)) for a in (70.0, 80.0, 85.0, 89.0, 90.0))
    assert r80 > 0.15, f"R(80)={r80} not on steep branch"
    assert r70 < r80 < r85 < r89, "high-AOI branch must rise monotonically up to 89 deg"
    assert r85 > 0.3, f"R(85)={r85} too low on steep branch"
    assert r89 > 0.5, f"R(89)={r89} too low on steep branch"
    assert abs(r90 - 0.08) < 1e-12, f"R(90)={r90} != 0.08 (grazing clamp)"
    assert 0.0 < r90 <= 1.0, "reflectivity clipped to [0,1]"
    # Vectorized shapes + scalar alias.
    assert aoi_reflectivity(np.array([0.0, 45.0, 90.0])).shape == (3,)
    # Custom transition angle keeps the branch boundary at the requested value.
    r_custom = float(aoi_reflectivity(70.0, transition_deg=80.0))
    r_direct = 1.1977e-5 * 70.0 ** 2 - 9.5728e-4 * 70.0 + 4.410e-2
    assert abs(r_custom - r_direct) < 1e-12, "custom transition should select polynomial branch"
    print(f"OK AOI reflectivity: R(0)={r0:.4f}, R(40)={r_min:.4f}, "
          f"R(60)={r60:.4f}, R(80)={r80:.4f}, R(85)={r85:.4f}, R(89)={r89:.4f}, R(90)={r90:.4f}")


def test_aoi_reflectivity_glass_types():
    # SGHAT v6 Table 1 selectable glass fits. "smooth" (default, w/o ARC) must be
    # unchanged; "smooth_arc" pins known-answer values from the paper's fit
    # (polynomial y=1.473E-5 x2 - 9.6416E-4 x + 3.2395E-2, exponential
    # y=4.7464E-5 e^0.1051x). Table 2 slope errors are stored in GLASS_FITS for
    # future use only.
    from solar_glare.hazard import (aoi_reflectivity, GLASS_FITS, GLASS_TYPES,
                                    _AOI_DEFAULT_TRANSITION)
    import numpy as np

    # Forward-compat reference values of the smooth fit must be unchanged.
    for a, ref in ((0.0, 0.0441), (40.0, 0.0250), (60.0, 0.0298),
                   (80.0, 0.2185), (85.0, 0.3636), (89.0, 0.5466), (90.0, 0.08)):
        assert abs(float(aoi_reflectivity(a)) - ref) < 1e-4, \
            f"smooth-default R({a})={aoi_reflectivity(a)} != {ref} (default must stay smooth)"
    # Default aliases the explicit smooth glass.
    xs = np.linspace(0.0, 90.0, 181)
    assert np.array_equal(aoi_reflectivity(xs),
                         aoi_reflectivity(xs, glass="smooth")), \
        "default glass must equal glass='smooth'"

    # Smooth glass w/ ARC (known-answer pins from SGHAT v6 Table 1).
    for a, ref in ((0.0, 0.032395), (40.0, 0.017397), (60.0, 0.027573),
                   (80.0, 0.212771), (85.0, 0.359860), (89.0, 0.547913),
                   (90.0, 0.08)):
        v = float(aoi_reflectivity(a, glass="smooth_arc"))
        assert abs(v - ref) < 1e-4, f"smooth_arc R({a})={v} != {ref}"
    # ARC lowers normal-incidence reflectivity (~3.2% vs ~4.4%) but keeps the
    # grazing clip; stays monotone up to the 89-deg cutoff.
    assert float(aoi_reflectivity(0.0, glass="smooth_arc")) < \
        float(aoi_reflectivity(0.0)), "smooth_arc R(0) should be below plain glass"
    r70, r80, r85, r89 = (float(aoi_reflectivity(a, glass="smooth_arc"))
                          for a in (70.0, 80.0, 85.0, 89.0))
    assert r70 < r80 < r85 < r89, "smooth_arc high branch must rise monotonically"

    # Every SGHAT v6 Table 1 type loads, stays in [0,1], and is monotone on the
    # exponential branch up to 89 deg; unknown keys raise.
    for g in GLASS_TYPES:
        assert g in GLASS_FITS and len(GLASS_FITS[g][0]) == 3 and \
            len(GLASS_FITS[g][1]) == 2 and len(GLASS_FITS[g][2:]) == 1, \
            f"GLASS_FITS[{g}] malformed (need poly3 + exp2 + slope_err)"
        vals = aoi_reflectivity(xs, glass=g)
        assert float(vals.min()) >= 0.0 and float(vals.max()) <= 1.0, \
            f"glass={g} outside [0,1]"
        hi = vals[(xs > _AOI_DEFAULT_TRANSITION) & (xs <= 89.0)]
        if hi.size:
            assert (np.diff(hi) >= -1e-12).all(), \
                f"glass={g} not monotone on high-AOI branch (trans,89]"
    try:
        aoi_reflectivity(45.0, glass="bogus")
        raise AssertionError("unknown glass type must raise ValueError")
    except ValueError:
        pass
    print("OK AOI reflectivity glass types: "
          + ", ".join(f"{g}(mrad={GLASS_FITS[g][2]})" for g in sorted(GLASS_FITS)))


def test_reflectivity_mode_runs():
    # End-to-end: reflectivity_mode="solar_weighted" (the pipeline default)
    # produces detail rows carrying the E903-20-style spectral reflectivity
    # (not the configured constant), and the "constant" path still stores the
    # configured value.
    import pandas as pd
    from solar_glare.analysis import _analyze_single

    class FakeArray:
        def __init__(self, normal):
            self.normal = normal
            self.char_size = 10.0
            self.specular_reflectivity = 0.08
            self.center_enu = np.array([0.0, 0.0, 0.0])

    class FakeDni:
        def __init__(self, n=1):
            self._series = pd.Series([1000.0] * n)
        @property
        def iloc(self):
            return self._series.iloc
        def to_numpy(self):
            return self._series.to_numpy()

    class FakeReceptor:
        def __init__(self, enu):
            self.enu = np.array(enu, dtype=float)

    def fake_pos(az, el):
        return pd.DataFrame(
            {"azimuth": np.array([az]), "apparent_elevation": np.array([el])},
            index=np.array([0]))

    # South-facing 45-degree panel; sun due south at 45 deg -> normal incidence,
    # reflected straight back at the sun. Place the receptor behind the panel
    # along ->r so the beam is on-axis.
    arr = FakeArray(np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0))
    s = sun_vector(180.0, 45.0)
    r_ref = reflect(s, arr.normal)

    # solar_weighted "smooth" R(0) ~4.18% (E903 spectral model), constant 0.08.
    for mode, expected in (("constant", 0.08), ("solar_weighted", 0.0418)):
        codes, det = _analyze_single(
            arr, FakeReceptor(r_ref * 2000.0), fake_pos(180.0, 45.0), FakeDni(1),
            reflectivity_mode=mode)
        assert codes[0] != "N" and len(det) == 1, f"{mode}: expected a glare hit"
        rho = float(det["reflectivity_used"].iloc[0])
        assert abs(rho - expected) < 1e-3, f"{mode}: rho={rho} != {expected}"
        aoi = float(det["aoi_deg"].iloc[0])
        assert abs(aoi - 0.0) < 1e-6, f"{mode}: aoi={aoi} != 0"
        assert (det["corneal_wcm2"].iloc[0]
                == det["dni_wm2"].iloc[0] / 1e4 * rho), "E_c must use stored rho"
    print("OK reflectivity end-to-end: constant rho=0.08, "
          "solar_weighted rho~0.0418 @ normal incidence")


def test_e903_solar_weighted():
    # ASTM E903-20-style solar-weighted reflectance (hazard.py): physical
    # Fresnel model of the cover glass solar-weighted by the IEC 60904-3:2019
    # direct-normal spectrum over [320, 2500] nm. Single-interface n~1.52 gives
    # R(0) ~4.2% (close to the SGHAT smooth 4.41%), grazing Fresnel rises
    # toward 1 (saturated at 89 deg, NO legacy 0.08 clip), and ARC lowers
    # normal-incidence reflectance.
    n550 = float(glass_refractive_index(550.0))
    assert abs(n550 - 1.5185) < 1e-3, f"n(550)={n550}"
    assert float(glass_refractive_index(320.0)) > \
        float(glass_refractive_index(2500.0)), "dispersion wrong way"

    r0 = float(solar_weighted_reflectivity(0.0, glass="smooth"))
    assert 0.03 < r0 < 0.06, f"R(0) smooth={r0}"
    r89 = float(solar_weighted_reflectivity(89.0, glass="smooth"))
    r90 = float(solar_weighted_reflectivity(90.0, glass="smooth"))
    assert r89 > 0.8, f"R(89)={r89} should be high (Fresnel, not 0.08 clip)"
    assert abs(r90 - r89) < 1e-9, "AOI >= 89 must saturate at R(89)"

    a0 = float(solar_weighted_reflectivity(0.0, glass="smooth_arc"))
    a80 = float(solar_weighted_reflectivity(80.0, glass="smooth_arc"))
    assert a0 < r0, "ARC must lower normal-incidence reflectance"
    assert a80 > a0, "grazing must rise above normal for ARC"

    for g in sorted(_E903_GLASS_TYPES):
        xs = np.linspace(0.0, 90.0, 181)
        v = solar_weighted_reflectivity(xs, glass=g)
        assert v.shape == xs.shape
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0, f"{g} out of [0,1]"
        sc = solar_weighted_reflectivity(40.0, glass=g)
        assert isinstance(sc, float) and 0.0 <= sc <= 1.0, f"{g} scalar alias"
    try:
        solar_weighted_reflectivity(10.0, glass="bogus")
        raise AssertionError("unknown glass must raise ValueError")
    except ValueError:
        pass
    print("OK E903 solar-weighted Fresnel: "
          + ", ".join(f"{g}: R(0)={float(solar_weighted_reflectivity(0.0, glass=g)):.4f}, "
                      f"R(89)={float(solar_weighted_reflectivity(89.0, glass=g)):.4f}"
                      for g in sorted(_E903_GLASS_TYPES)))


def test_glare_polygon_hit():
    """True-rectangle polyhedral emanation (glare_polygon_hit) must:
      (a) match the area-equivalent-circle circular cone for far small
          panels (circle is the correct limit there, near-square or
          moderately elongated);
      (b) match the sampled exact rectangle geometry for an elongated
          panel where the circular cone distorts (under long / over short);
      (c) honour the beam-spread half-angle boundary (beta/2).
    """
    from solar_glare.hazard import beam_spread_angle, array_limiting_angle

    n = np.array([0.0, 0.0, 1.0])          # horizontal panel, normal up
    hb = beam_spread_angle(0.0) / 2.0      # theta_sun/2 = 4.65 mrad

    def sun_swath(az0, el0, daz, del_el, m=11):
        az = np.linspace(az0 - daz, az0 + daz, m)
        el = np.linspace(max(el0 - del_el, 0.5), el0 + del_el, m)
        aa, ee = np.meshgrid(az, el, indexing="ij")
        return sun_vector(aa.ravel(), ee.ravel())

    def beam_receptor(az0, el0, d):
        r = reflect(sun_vector(az0, el0), n)
        # receptor along the reflected direction, tilted slightly above plane
        return r * d + np.array([0.0, 0.0, 1.5])

    def sampled_hits(sv, corners, recp):
        """(J-point) sampled exact: any sample point P on the rectangle with
        angle(U-P, r) <= beta/2."""
        e, s_ax = panel_axes(n)
        cc = np.mean(corners, axis=0)
        ww = np.linalg.norm(corners[0] - corners[1])
        hh = np.linalg.norm(corners[0] - corners[3])
        u = np.linspace(-0.5, 0.5, 25)
        v = np.linspace(-0.5, 0.5, 25)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        P = cc + (uu[..., None] * e) * ww + (vv[..., None] * s_ax) * hh
        P = P.reshape(-1, 3)
        uv = recp - P
        uv = uv / np.linalg.norm(uv, axis=-1, keepdims=True)   # (J,3)
        rv = reflect(sv, n)                                     # (N,3)
        ang = np.arccos(np.clip(uv @ rv.T, -1.0, 1.0))          # (J,N)
        return (ang <= hb).any(axis=0)

    def circular_hits(sv, corners, recp):
        """SGHAT v6 classic: U->receptor within theta/2 + beta/2 of r."""
        ww = np.linalg.norm(corners[0] - corners[1])
        hh = np.linalg.norm(corners[0] - corners[3])
        u_vec = recp
        d = np.linalg.norm(u_vec)
        u_hat = u_vec / d
        r_vec = reflect(sv, n)
        th = array_limiting_angle(ww * hh, d, np.abs(r_vec @ n))
        return angle_between(r_vec, u_hat) <= th / 2.0 + hb

    # --- (a) far small panels (near-square and elongated): all three agree ---
    for w, h, d in ((20.0, 20.0, 3000.0), (10.0, 40.0, 5000.0)):
        corners = panel_corners_enu(np.zeros(3), n, w, h)
        s = sun_swath(90.0, 15.0, 8.0, 2.5)
        recp = beam_receptor(90.0, 15.0, d)
        poly = glare_polygon_hit(s, corners, recp, n, hb)
        smp = sampled_hits(s, corners, recp)
        circ = circular_hits(s, corners, recp)
        assert float((poly == smp).mean()) >= 0.999, "near-square polygon != sampled"
        assert float((poly == circ).mean()) >= 0.999, "near-square polygon != circular"
    print("OK polygon hit (a) far near-square: polygon == circular == sampled")

    # --- (b) elongated 500x20 m @ 1500 m: polygon == sampled; circular distorts ---
    w, h, d = 500.0, 20.0, 1500.0
    corners = panel_corners_enu(np.zeros(3), n, w, h)
    s = sun_swath(90.0, 15.0, 12.0, 3.0)
    recp = beam_receptor(90.0, 15.0, d)
    poly = glare_polygon_hit(s, corners, recp, n, hb)
    smp = sampled_hits(s, corners, recp)
    circ = circular_hits(s, corners, recp)
    agree_p = float((poly == smp).mean())
    agree_c = float((poly == circ).mean())
    assert agree_p >= 0.999, f"elongated polygon vs sampled agree={agree_p:.5f}"
    assert agree_c < 0.999, "elongated circular should NOT match polygon (distortion)"
    assert int(poly.sum()) > int(circ.sum()), \
        "elongated true footprint should exceed circular (circular under-predicts long axis)"
    diff = int(np.count_nonzero(circ != poly))
    print(f"OK polygon hit (b) elongated 500x20 m: poly==sampled agree={agree_p:.5f}, "
          f"polygon hits={int(poly.sum())} vs circular={int(circ.sum())} "
          f"({diff} cells differ -> circular distortion is real)")

    # --- (c) beam-spread boundary: hit <= beta/2, miss beyond ---
    w, h, d = 1.0, 1.0, 2000.0
    corners = panel_corners_enu(np.zeros(3), n, w, h)
    recp = beam_receptor(90.0, 15.0, d)
    s0 = sun_vector(90.0, 15.0)
    devs = np.array([0.0, hb / 3.0, hb, hb * 2.0, hb * 3.0])
    hits = []
    for dd in devs:
        c, si = math.cos(dd), math.sin(dd)
        s_rot = np.array([s0[0] * c - s0[1] * si, s0[0] * si + s0[1] * c, s0[2]])
        hits.append(bool(glare_polygon_hit(s_rot, corners, recp, n, hb)))
    assert hits == [True, True, True, False, False], \
        f"beam-spread fan {np.round(devs*1e3, 2)} mrad -> {hits} (expect 3 hits then miss)"
    print(f"OK polygon hit (c) beam-spread boundary: {np.round(devs*1e3, 2)} mrad -> {hits}")


def test_beer_lambert_attenuation():
    # Known answer from references/Atmospheric_Optical_Attenuation_Model_ver2.txt:
    # V=20 km, lambda=550 nm, d=1 km -> sigma=0.1955 km^-1, tau~0.8224.
    sigma = extinction_coefficient(20.0, 550.0)
    approx(sigma, 0.1955, 1e-3, "sigma(20km,550nm)")
    tau = atmospheric_transmittance(1000.0, 20.0, 550.0)
    approx(tau, math.exp(-0.1955), 1e-3, "tau(1km,20km) ~ 0.8224")
    approx(tau, 0.8224, 1e-3, "tau(1km,20km) vs doc")

    # Bounds and regime QA.
    assert atmospheric_transmittance(0.0, 20.0) == 1.0, "tau(d=0) must be 1"
    assert 0.0 < atmospheric_transmittance(50000.0, 20.0) < 1.0, "tau in (0,1)"
    # q regime switching: >50km -> 1.6, (6,50] -> 1.3, (1.2,6] -> 0.585 V^(1/3),
    # <=1.2km -> 0 (so sigma = 3.91/V, pure geometric thinning in fog).
    s_clear = extinction_coefficient(60.0)
    s_std = extinction_coefficient(20.0)
    s_haze = extinction_coefficient(4.0)
    s_fog = extinction_coefficient(1.0)
    assert s_haze > s_std > 0 and s_std > s_clear > 0, \
        f"extinction ordering clear<haze: {s_clear:.3f}/{s_std:.3f}/{s_haze:.3f}/{s_fog:.3f}"
    # Fog q=0 -> sigma = 3.91/V (independent of lambda).
    approx(s_fog, 3.91 / 1.0, 1e-9, "fog sigma = 3.91/V")
    # lambda dependence: shorter wavelength (blue) attenuates MORE than 550nm
    # under q>0 (sigma ~ lambda^-q), so tau(450nm) < tau(650nm).
    t_blue = atmospheric_transmittance(5000.0, 20.0, 450.0)
    t_red = atmospheric_transmittance(5000.0, 20.0, 650.0)
    assert t_blue < t_red, "blue attenuates more than red under Kruse q>0"
    print("OK Beer-Lambert attenuation (Kruse-Kim known-answer + regime bounds)")


if __name__ == "__main__":
    test_mirror_law()
    test_normal_incidence()
    test_known_panel_geometry()
    test_retinal_irradiance_formula()
    test_threshold_shape()
    test_classify_zones()
    test_sun_position_spa_eot()
    test_heatmap_matrix_shape()
    test_array_limiting_angle()
    test_front_face_sun_disc_guard()
    test_aoi_reflectivity()
    test_aoi_reflectivity_glass_types()
    test_reflectivity_mode_runs()
    test_e903_solar_weighted()
    test_glare_polygon_hit()
    test_beer_lambert_attenuation()
    print("ALL SANITY CHECKS PASSED")
