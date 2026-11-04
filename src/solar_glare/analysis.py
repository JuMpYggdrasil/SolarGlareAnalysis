"""Site model and full-year glare analysis at configurable time resolution
(data-driven from config/site.yaml)."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time as _time

import numpy as np
import pandas as pd
import pvlib
import yaml

from .geometry import (enu_from_latlon, sun_vector, surface_normal, reflect,
                       angle_between, panel_corners_enu, glare_polygon_hit,
                       _panel_angular_span)
from .hazard import (
    corneal_irradiance,
    retinal_irradiance,
    classify,
    array_limiting_angle,
    beam_spread_angle,
    effective_subtended_angle,
    solar_weighted_reflectivity,
    atmospheric_transmittance,
    _AOI_DEFAULT_GLASS,
    _E903_GLASS_TYPES,
)
from .iec62471 import evaluate_glare as iec_evaluate_glare
from .solar import build_times, solar_position, clearsky_dni
from .profiling import TIMER, stage_fn


@dataclass
class PVArray:
    name: str
    lat: float
    lon: float
    height_agl_m: float
    width_m: float
    height_m: float
    tilt_deg: float
    azimuth_deg: float
    specular_reflectivity: float
    center_enu: np.ndarray = field(init=False)
    groups: int = 1
    group_arrays: list = field(default_factory=list, init=False, repr=False)

    def setup(self, ref_lat, ref_lon):
        with TIMER.stage("PVArray.setup"):
            self.center_enu = enu_from_latlon(ref_lat, ref_lon, self.lat, self.lon, self.height_agl_m)
            self.normal = surface_normal(self.tilt_deg, self.azimuth_deg)
            # characteristic source size used for the subtended angle (m)
            self.char_size = np.sqrt(self.width_m * self.height_m)
            self.area = self.width_m * self.height_m
            self.hazard_area = self.area   # SGHAT §5.5 source = full contiguous sheet
            # split the panel height into `groups` sub-panels for finer near-field
            # detection; the parent array still drives all display/aggregation.
            self.group_arrays = self._split_groups(ref_lat, ref_lon)

    def _split_groups(self, ref_lat, ref_lon):
        """Sub-divide the array height into `groups` equal sub-panels.

        Each sub-panel keeps the parent's width, orientation and reflectivity
        but has height_m/groups and a center offset along the panel's height
        axis s = cross(n, e) (same in-plane height direction used by
        viz3d._panel_corners, so the group footprints tile the parent exactly).
        """
        n = max(int(self.groups), 1)
        if n <= 1:
            return [self]
        with TIMER.stage("PVArray._split_groups") as rec:
            up = np.array([0.0, 0.0, 1.0])
            e = up - self.normal * np.dot(up, self.normal)  # width axis in panel plane
            if np.linalg.norm(e) < 1e-9:                     # horizontal panel
                e = np.array([1.0, 0.0, 0.0])
            else:
                e = e / np.linalg.norm(e)
            s = np.cross(self.normal, e)                     # height axis in panel plane
            gh = self.height_m / n
            rec.key(subpanel_height=gh, subpanel_area=self.width_m * gh)
            groups = []
            for i in range(n):
                off = (i - (n - 1) / 2.0) * gh
                g = PVArray(
                    name=f"{self.name}#{i}",
                    lat=self.lat, lon=self.lon,
                    height_agl_m=self.height_agl_m,
                    width_m=self.width_m, height_m=gh,
                    tilt_deg=self.tilt_deg, azimuth_deg=self.azimuth_deg,
                    specular_reflectivity=self.specular_reflectivity,
                    groups=1,
                )
                g.center_enu = self.center_enu + s * off
                g.normal = self.normal
                g.char_size = np.sqrt(self.width_m * gh)
                g.area = self.width_m * gh
                g.hazard_area = self.width_m * self.height_m   # full contiguous sheet (SGHAT §5.5)
                g.group_arrays = []
                groups.append(g)
            return groups


@dataclass
class Receptor:
    name: str
    lat: float
    lon: float
    height_agl_m: float
    enu: np.ndarray = field(init=False)

    def setup(self, ref_lat, ref_lon):
        with TIMER.stage("Receptor.setup"):
            self.enu = enu_from_latlon(ref_lat, ref_lon, self.lat, self.lon, self.height_agl_m)


def load_config(path="config/site.yaml"):
    with TIMER.stage("load_config") as rec:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cases = (cfg.get("site") or {}).get("cases") or []
        rec.key(n_cases=len(cases))
        return cfg


def build_site(cfg):
    with TIMER.stage("build_site") as rec:
        site = cfg["site"]
        ref = site["reference"]
        arrays = [PVArray(**a) for a in cfg["arrays"]]
        receptors = [Receptor(**r) for r in cfg["receptors"]]
        for a in arrays:
            a.setup(ref["lat"], ref["lon"])
        for r in receptors:
            r.setup(ref["lat"], ref["lon"])
        rec.key(n_arrays=len(arrays), n_receptors=len(receptors),
                n_groups=(arrays[0].groups if arrays else 0))
        return site, arrays, receptors


def compute_ephemeris(cfg, site):
    """Return solar position + DNI for the full year at the requested
    resolution (``site.time_freq``: "h" -> 8,760 steps, "min" -> 525,600).

    DNI is either the pvlib Ineichen clear-sky model (``environment.dni_mode ==
    "ineichen"``, the default, physically-based) or a fixed value
    (``environment.dni_mode == "constant"`` + ``environment.dni_constant_wm2``)
    matching the SGHAT/ForgeSolar rule-of-thumb assumption of a constant
    1000 W/m2 DNI - used ONLY for apples-to-apples benchmarking against
    published SGHAT results. Detection (sun position) is unchanged.
    """
    with TIMER.stage("compute_ephemeris") as rec:
        ref = site["reference"]
        freq = site.get("time_freq", "h")
        with TIMER.stage("build_times") as trec:
            times = build_times(int(site["year"]), site.get("timezone", "Asia/Bangkok"), freq=freq)
            trec.key(n_timesteps=len(times),
                     time_step=(float((times[1] - times[0]).total_seconds())
                                if len(times) > 1 else None))
        loc = pvlib.location.Location(
            ref["lat"], ref["lon"], tz=site.get("timezone", "Asia/Bangkok"),
            altitude=ref.get("height_m", 0.0),
        )
        with TIMER.stage("solar_position"):
            pos = solar_position(times, ref["lat"], ref["lon"], altitude=ref.get("height_m", 0.0))
        env = cfg.get("environment", {})
        mode = env.get("dni_mode", "ineichen")
        with TIMER.stage("clearsky_dni"):
            if mode == "constant":
                dni = pd.Series(float(env.get("dni_constant_wm2", 1000.0)), index=times)
            else:
                dni = clearsky_dni(loc, times, site.get("linke_turbidity", 3.5))
        rec.key(n_timesteps=len(times))
        return times, pos, dni


_HAZARD_RANK = {"N": 0, "G": 1, "Y": 2, "R": 3}


@stage_fn("_rank_codes")
def _rank_codes(codes):
    """Map hazard letters to rank ints (N=0 < G=1 < Y=2 < R=3) for merging."""
    ranks = np.zeros(np.size(codes), dtype=int)
    ranks[codes == "G"] = 1
    ranks[codes == "Y"] = 2
    ranks[codes == "R"] = 3
    return ranks


@stage_fn("_codes_from_rank")
def _codes_from_rank(ranks):
    """Inverse of _rank_codes."""
    codes = np.full(np.shape(ranks), "N", dtype="<U1")
    codes[ranks == 1] = "G"
    codes[ranks == 2] = "Y"
    codes[ranks == 3] = "R"
    return codes


@stage_fn("_dedupe_detail")
def _dedupe_detail(detail):
    """One row per glare minute: keep the worst hazard rank (tie-broken by the
    highest retinal irradiance) so per-pair detail matches the worst-wins merge
    used for the summary. `group` retains which sub-panel produced the winner."""
    if detail.empty:
        return detail
    rank = detail["hazard"].map(_HAZARD_RANK)
    detail = detail.assign(_rank=rank)
    detail = (detail.sort_values(["_rank", "retinal_wcm2"], ascending=[False, False])
              .drop_duplicates(subset=["time"], keep="first")
              .drop(columns="_rank")
              .sort_values("time")
              .reset_index(drop=True))
    return detail


@stage_fn("_analyze_single")
def _analyze_single(array, receptor, pos, dni, slope_error_rad=0.0, min_dni_wm2=0.0,
                    reflectivity_mode="constant",
                    glass_type=_AOI_DEFAULT_GLASS, hit_test_mode="circular",
                    iec_enabled=False, solar_weighted_n_arc=None,
                    solar_weighted_arc_wl0=600.0,
                    attenuation_mode="none", visibility_km=20.0,
                    attenuation_wl_nm=550.0):
    """Glare status for one (array, receptor) pair over the full year.

    `min_dni_wm2` > 0 drops glare minutes whose DNI falls below the floor
    (config `environment.min_dni_wm2`); 0.0 disables the filter.

    `reflectivity_mode` selects the surface reflectivity: "solar_weighted"
    (the pipeline default, per config `environment.reflectivity_mode`) uses
    the ASTM E903-20-style solar-weighted reflectance
    (`solar_weighted_reflectivity`, Fresnel spectral model weighted by the
    IEC 60904-3:2019 Table 1 direct-normal spectrum; `glass_type`
    "smooth"/"smooth_arc", `solar_weighted_n_arc`/`solar_weighted_arc_wl0`
    tune the optional ARC layer); "constant" uses the array's configured
    `specular_reflectivity` at all angles.

    `hit_test_mode` selects the glare-hit detection geometry:
      "circular" (default): SGHAT v6 classic - a receptor is hit when the
        reflected sun direction lies within the beam half-angle
        `θ/2 + β/2`, where θ is the *area-equivalent circle* array-limiting
        angle sqrt(4·A·|cos|/π)/d.  Fast, matches published SGHAT results,
        but over-predicts across a panel's short axis / under-predicts along
        its long axis for elongated rows.
      "polygon": true-rectangle "polyhedral emanation" - the 4 panel corners
        are projected onto the receiver's viewing sphere, reflected about the
        panel normal into specular-sun space, and the sun is tested against
        the resulting spherical convex polygon dilated by β/2.  Exactly the
        analytic equivalent of sampling the real rectangle (SGHAT Sec.4.4);
        no area-equivalent-circle distortion.  Retinal hazard x stays
        `min(β, θ)` either way (detection only changes).

    Returns (codes, detail) where codes is an array of 'N'/'G'/'Y'/'R' over all
    timesteps, and detail is a DataFrame for the glare timesteps with geometry.
    """
    n = len(pos)
    codes = np.full(n, "N", dtype="<U1")

    s = sun_vector(pos["azimuth"].to_numpy(), pos["apparent_elevation"].to_numpy())
    u_vec = receptor.enu - array.center_enu
    dist = np.linalg.norm(u_vec)
    u_hat = u_vec / dist

    r_vec = reflect(s, array.normal)
    r_dot_u = (r_vec * u_hat[None, :]).sum(axis=1)
    theta = angle_between(r_vec, u_hat[None, :])

    s_dot_n = (s * array.normal[None, :]).sum(axis=1)
    # Front-face illumination: require the WHOLE sun disc (half-angle beta/2 =
    # theta_sun/2 + 6*sigma) to sit above the panel plane. When the sun grazes
    # the plane (s.n < sin(beta/2)) the front/back of the panel is ambiguous and
    # part of the reflected cone would appear to come from behind the array, so
    # such hits are rejected. (SGHAT v6 documents no such gate; this is a
    # conservative guard against the degenerate low-sun case for near-horizontal
    # panels, e.g. tilt ~0 with the sun on the horizon.)
    half_spread = beam_spread_angle(slope_error_rad) / 2.0
    front = s_dot_n > np.sin(half_spread)
    ahead = r_dot_u > 0.0
    # The reflected beam from a flat panel is collimated with transverse width
    # equal to the panel size; the receptor sees glare when it lies inside that
    # beam. The half-angle accounts for the array's *visible* angular extent
    # (SGHAT v6 Sec.5.5 array-limiting angle, which foreshortens when the array
    # is viewed from the side) plus the reflected beam's own half-angle
    # (beta/2 = theta_sun/2 + 6*sigma). REF: SGHAT v6 Sec.4.3 + Sec.5.5; the
    # full cone-plane hit test of Sec.4.4 is approximated by this collimated-
    # beam half-angle.
    area = getattr(array, "area", array.char_size ** 2)
    hazard_area = getattr(array, "hazard_area", area)
    abs_cos = np.abs((r_vec * array.normal[None, :]).sum(axis=1))
    if hit_test_mode == "polygon":
        # True-rectangle polyhedral emanation: the 4 panel corners projected
        # onto the receiver's viewing sphere, reflected into specular-sun
        # space, dilated by beta/2.  No area-equivalent-circle distortion.
        corners = panel_corners_enu(array.center_enu, array.normal,
                                    array.width_m, array.height_m)
        # Cheap conservative prefilter (vectorized): any glare requires
        # the reflected direction r to lie within the panel's own angular
        # footprint + beta/2 of the receiver direction, so rows far off-beam
        # can be rejected before paying the O(N)-per-pair polygon test.
        span = _panel_angular_span(corners, receptor.enu, u_hat)
        pre = front & ahead & (theta <= span + half_spread)
        poly_hit = glare_polygon_hit(s, corners, receptor.enu, array.normal,
                                     half_spread, subset=pre)
        glare_mask = poly_hit & (pos["apparent_elevation"] > 0)
    else:
        # SGHAT v6 classic: half-angle = array-limiting (visible) half-angle +
        # beam-spread half-angle (beta/2 = theta_sun/2 + 6*sigma). The full
        # cone-plane hit test (Sec.4.4) is approximated by this collimated
        # beam half-angle of the area-equivalent circle.
        beam_half = array_limiting_angle(area, dist, abs_cos) / 2.0 + half_spread
        glare_mask = front & ahead & (theta <= beam_half) & (pos["apparent_elevation"] > 0)
    if min_dni_wm2 > 0.0:
        # Optional minimum-DNI floor (config environment.min_dni_wm2): drops
        # glare minutes where essentially no sunlight reaches the array (e.g.
        # sun on the horizon, where the refraction-corrected DNI model is not
        # physically meaningful).
        glare_mask &= dni.to_numpy() > min_dni_wm2

    idx = np.where(glare_mask)[0]
    if idx.size == 0:
        return codes, pd.DataFrame()

    # REF: SGHAT v6 Sec.5.5 subtended source angle x = min(beta, array-limiting)
    # used in Eq.(1)-(2) for corneal/retinal irradiance; Fig.2 for G/Y/R zones.
    # x uses the FULL contiguous reflective surface (hazard_area): sub-paneling
    # resolves the detection footprint but does NOT split the physical source a
    # coplanar mirror presents to the eye (the image stays sun-limited, x=beta,
    # whenever the whole sheet subtends > beta). Using the sub-panel area here
    # would shrink x with finer groups -> the "one-pixel mirror" Red artifact.
    x = effective_subtended_angle(hazard_area, dist, abs_cos[idx], slope_error_rad)
    # Angle of incidence (deg) = angle between sun vector and panel normal.
    # The front-face gate keeps s.n > 0, so aoi is in [0, 90).
    aoi_deg = np.degrees(np.arccos(np.clip(s_dot_n[idx], 0.0, 1.0)))
    if reflectivity_mode == "solar_weighted":
        rho = solar_weighted_reflectivity(
            aoi_deg, glass=glass_type,
            n_arc=solar_weighted_n_arc,
            arc_design_wl_nm=solar_weighted_arc_wl0)
    else:
        rho = array.specular_reflectivity
    tau = None
    if attenuation_mode == "beer_lambert":
        tau = atmospheric_transmittance(dist, visibility_km, attenuation_wl_nm)
    ec = corneal_irradiance(dni.iloc[idx].to_numpy(), rho)
    if tau is not None:
        ec = ec * tau
    er = retinal_irradiance(ec, x)
    hz = classify(er, x)
    codes[idx] = hz

    detail = pd.DataFrame({
        "time": pos.index[idx],
        "group": getattr(array, "name", "panel"),
        "elevation_deg": pos["apparent_elevation"].iloc[idx].to_numpy(),
        "azimuth_deg": pos["azimuth"].iloc[idx].to_numpy(),
        "dni_wm2": dni.iloc[idx].to_numpy(),
        "aoi_deg": aoi_deg,
        "reflectivity_used": rho,
        "theta_rad": theta[idx],
        "subtended_rad": np.asarray(x),
        "atmos_transmittance": np.full(idx.size, float(tau)) if tau is not None else np.full(idx.size, 1.0),
        "corneal_wcm2": ec,
        "retinal_wcm2": er,
        "hazard": hz,
    })
    if iec_enabled:
        # IEC 62471 columns built from the SAME ec/er the SGHAT pipeline just
        # computed (radiance from E_c, R-weighted retinal stays K_EYE-consistent
        # with the radiance form). `width_m` is the per-sub-panel width; the
        # full contiguous sheet drives hazard, so alpha uses the parent width
        # when the group model split height only.
        iw = array.width_m
        ih = max(getattr(array, "hazard_area", array.width_m * array.height_m), 1.0) / max(array.width_m, 1e-9)
        r = iec_evaluate_glare(ec, er, dist, iw, ih,
                               abs_cos_ref=abs_cos[idx])
        for col in r:
            detail[f"iec_{col}"] = r[col]
    return codes, detail


def analyze_pair(array, receptor, pos, dni, slope_error_rad=0.0, min_dni_wm2=0.0,
                 reflectivity_mode="constant",
                 glass_type=_AOI_DEFAULT_GLASS, n_workers=None,
                 hit_test_mode="circular", iec_enabled=False,
                 solar_weighted_n_arc=None, solar_weighted_arc_wl0=600.0,
                 attenuation_mode="none", visibility_km=20.0,
                 attenuation_wl_nm=550.0):
    """Glare status for one (array, receptor) pair over the full year.

    If the array is split into sub-groups (`groups > 1` in config), each
    sub-panel is analysed with its own centre/area and the results are merged:
    per timestep the worst hazard level wins (N < G < Y < R), and the returned
    detail keeps one row per glare minute (worst hazard, tie-broken by highest
    retinal irradiance) with a `group` column naming the contributing sub-panel.
    The caller only ever sees the parent array, so display/aggregation is
    unaffected.

    ``n_workers`` (>=1) runs the independent sub-panel analyses on a shared
    thread pool. Each ``_analyze_single`` is numpy-vectorised over the whole
    year (releases the GIL), so threads give near-linear speedup on the group
    loop without pickling the large pos/dni tables. Values: None or 0 = auto
    (``min(cpu_count, groups)``), 1 = serial, N = fixed thread count. For
    ``groups <= 1`` the serial path is used regardless. The worst-wins merge is
    commutative, so the result is bit-identical to the serial path.
    """
    groups = getattr(array, "group_arrays", None) or []
    if len(groups) <= 1:
        with TIMER.stage("analyze_pair") as rec:
            rec.parallel(tasks=1, workers=1, peak=1)
            return _analyze_single(array, receptor, pos, dni, slope_error_rad, min_dni_wm2,
                                   reflectivity_mode, glass_type,
                                   hit_test_mode, iec_enabled,
                                   solar_weighted_n_arc, solar_weighted_arc_wl0,
                                   attenuation_mode, visibility_km, attenuation_wl_nm)

    if n_workers is None or n_workers == 0:
        n_workers = min(os.cpu_count() or 1, len(groups))
    n_workers = max(int(n_workers), 1)

    with TIMER.stage("analyze_pair") as rec:
        t_pair0 = _time.perf_counter()
        # `tids` counts the distinct threads that actually run a sub-panel, and
        # `cpu_equiv_s` is the summed sub-panel wall time, so the speedup is
        # measured (serial-equivalent work / actual wall), not assumed.
        # `peak` is the true maximum number of sub-panels in flight at once.
        tids = set()
        cpu_equiv_s = 0.0
        live = [0, 0]                     # [current in flight, peak]
        live_lock = threading.Lock()

        def _one(g, idx):
            """Run one sub-panel, recording its thread + span for the lane
            chart. Returns the untouched _analyze_single result."""
            nonlocal cpu_equiv_s
            ts = _time.perf_counter()
            tid = threading.get_ident()
            with live_lock:
                tids.add(tid)
                live[0] += 1
                if live[0] > live[1]:
                    live[1] = live[0]
            try:
                out = _analyze_single(g, receptor, pos, dni, slope_error_rad, min_dni_wm2,
                                      reflectivity_mode, glass_type,
                                      hit_test_mode, iec_enabled,
                                      solar_weighted_n_arc, solar_weighted_arc_wl0,
                                      attenuation_mode, visibility_km, attenuation_wl_nm)
            finally:
                with live_lock:
                    live[0] -= 1
            te = _time.perf_counter()
            cpu_equiv_s += te - ts
            TIMER.task_span(tid, ts, te, idx)
            return out

        if n_workers == 1 or len(groups) == 1:
            results = [_one(g, i) for i, g in enumerate(groups)]
        else:
            # _one closes over every per-pair argument, so it can be mapped
            # directly; map() walks its iterables in lockstep and preserves
            # order, which the worst-wins merge below relies on.
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_one, groups, range(len(groups))))

        rank = None
        details = []
        for codes, detail in results:
            r = _rank_codes(codes)
            rank = r if rank is None else np.maximum(rank, r)
            if not detail.empty:
                details.append(detail)
        merged = _codes_from_rank(rank)
        if details:
            detail = pd.concat(details, ignore_index=True)
            merged, detail = merged, _dedupe_detail(detail)
        else:
            detail = pd.DataFrame()

        pair_wall = _time.perf_counter() - t_pair0
        n_run = max(len(tids), 1)
        rec.parallel(tasks=len(groups), workers=n_run, peak=live[1])
        rec.key(parallel_tasks=len(groups), parallel_threads=n_run,
                peak_concurrency=live[1],
                n_task_spans=len(groups),
                speedup_vs_serial=(cpu_equiv_s / pair_wall
                                   if pair_wall > 0 and cpu_equiv_s > 0 else None))
        return merged, detail


def run_analysis(cfg, hour_range=None, progress=False):
    """Full-year analysis over all arrays x receptors at the configured
    time resolution (time_freq in config/site.yaml).

    Returns (times, pos, dni, long_df, summary_df). long_df keeps only glare rows
    (hazard != "N") so sub-hourly resolutions do not blow up memory.

    If hour_range is provided as (start_hour, end_hour) inclusive (0-23),
    the analysis is limited to that hour window (e.g., (7, 8) = 07:00-08:59).

    progress=True prints `=== <receptor name>` before each pair is analysed
    (stdout only, no effect on results); callers default to silent.
    """
    with TIMER.stage("run_analysis") as rec:
        return _run_analysis_impl(cfg, rec, hour_range, progress)


def _run_analysis_impl(cfg, rec, hour_range=None, progress=False):
    site, arrays, receptors = build_site(cfg)
    times, pos, dni = compute_ephemeris(cfg, site)

    # Time range filter (inclusive: start_hour <= hour <= end_hour)
    if hour_range:
        h_start, h_end = hour_range
        mask = (times.hour >= h_start) & (times.hour <= h_end)
        times = times[mask]
        pos = pos.loc[mask]
        dni = dni.loc[mask]
        rec.key(n_timesteps=len(times), hour_range=list(hour_range))
    n_times = len(times)
    slope_err = cfg.get("environment", {}).get("slope_error_rad", 0.0)
    min_dni = cfg.get("environment", {}).get("min_dni_wm2", 10.0)
    refl_mode = cfg.get("environment", {}).get("reflectivity_mode", "solar_weighted")
    glass_type = cfg.get("environment", {}).get("glass_type", _AOI_DEFAULT_GLASS)
    if glass_type not in _E903_GLASS_TYPES:
        raise ValueError(
            f"environment.glass_type {glass_type!r} is not an "
            f"E903-model glass type; choose from {sorted(_E903_GLASS_TYPES)} "
            f"(textured fits are SGHAT v6 Table 1 reference-only)")
    sw_n_arc = cfg.get("environment", {}).get("solar_weighted_arc_n", None)
    sw_arc_wl0 = float(cfg.get("environment", {}).get(
        "solar_weighted_arc_design_wl_nm", 600.0))
    n_workers = cfg.get("environment", {}).get("n_workers", 0)
    hit_mode = cfg.get("environment", {}).get("hit_test_mode", "circular")
    iec_enabled = bool(cfg.get("environment", {}).get("iec62471_enabled", False))
    atten_mode = cfg.get("environment", {}).get("atmospheric_attenuation", "none")
    if atten_mode not in ("none", "beer_lambert"):
        raise ValueError(
            f"environment.atmospheric_attenuation {atten_mode!r} not in "
            f"('none','beer_lambert')")
    vis_km = float(cfg.get("environment", {}).get("visibility_km", 20.0))
    atten_wl = float(cfg.get("environment", {}).get(
        "attenuation_wavelength_nm", 550.0))

    rows = []
    details = {}
    sum_rows = []
    rec.key(n_pairs=len(arrays) * len(receptors))
    for a in arrays:
        for r in receptors:
            if progress:
                print(f"=== {r.name}", flush=True)
            key = (a.name, r.name)
            codes, detail = analyze_pair(a, r, pos, dni, slope_err, min_dni,
                                         refl_mode, glass_type,
                                         n_workers, hit_mode, iec_enabled,
                                         sw_n_arc, sw_arc_wl0,
                                         atten_mode, vis_km, atten_wl)
            details[key] = detail
            glare = codes != "N"
            rows.append(pd.DataFrame({
                "time": times[glare],
                "array": a.name,
                "receptor": r.name,
                "hazard": codes[glare],
            }))
            sum_rows.append({
                "receptor": r.name, "array": a.name,
                "N": int((codes == "N").sum()),
                "G": int((codes == "G").sum()),
                "Y": int((codes == "Y").sum()),
                "R": int((codes == "R").sum()),
            })

    long_df = pd.concat(rows, ignore_index=True)
    summary = (pd.DataFrame(sum_rows)
               .set_index(["receptor", "array"])
               .reindex(columns=["N", "G", "Y", "R"], fill_value=0))
    summary["minutes_with_glare"] = summary[["G", "Y", "R"]].sum(axis=1)
    summary["annual_peak_retinal_wcm2"] = np.nan
    summary["peak_subtended_rad"] = np.nan
    if iec_enabled:
        summary["iec_worst_risk"] = pd.Series(pd.NA, index=summary.index, dtype=object)
        summary["iec_min_tmax_thermal_s"] = np.nan
        summary["iec_min_tmax_blue_s"] = np.nan
        summary["iec_peak_lr_w_sr_m2"] = np.nan
    _IEC_RANK = {"Exempt": 0, "RG1": 1, "RG2": 2, "RG3": 3}
    for (a, r), d in details.items():
        if not d.empty:
            peak = d.loc[d["retinal_wcm2"].idxmax()]
            mask = (summary.index.get_level_values("array") == a) & (
                summary.index.get_level_values("receptor") == r
            )
            summary.loc[mask, "annual_peak_retinal_wcm2"] = peak["retinal_wcm2"]
            summary.loc[mask, "peak_subtended_rad"] = peak["subtended_rad"]
            if iec_enabled and "iec_risk_eye" in d.columns:
                worst = d.loc[d["iec_risk_eye"].map(_IEC_RANK).idxmax()]
                summary.loc[mask, "iec_worst_risk"] = worst["iec_risk_eye"]
                summary.loc[mask, "iec_min_tmax_thermal_s"] = float(
                    d["iec_tmax_thermal_s"].min())
                summary.loc[mask, "iec_min_tmax_blue_s"] = float(
                    d["iec_tmax_blue_s"].min())
                summary.loc[mask, "iec_peak_lr_w_sr_m2"] = float(
                    d["iec_lr_w_sr_m2"].max())

    return times, pos, dni, long_df, summary, details, site, arrays, receptors
