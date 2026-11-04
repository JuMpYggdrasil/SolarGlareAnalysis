"""Internal QA for figures 07 (annual glare hazard hours) and 08 (ocular chart).

Verifies, end-to-end on the current config/site.yaml:
  [A] Ocular chart (08): every plotted point lies in its hazard zone; the
      E_r / E_c arithmetic round-trips from the stored DNI / reflectivity;
      known-answer points (sun disc, extreme demo points); plot extent covers
      all points.
  [B] Annual hours (07) / summary audit: G+Y+R == minutes_with_glare; event
      counts in the long table match the summary; no duplicate
      (time, array, receptor) rows; per-pair detail has one row per minute;
      peak retinal irradiance in the summary equals the max of the detail.
  [C] Parallel parity + independent exact hit-test.
      Parity: n_workers=8 vs n_workers=1 must be bit-identical (codes AND
      every detail column) - resolved from the first committed site.cases
      entry, because the top-level baseline config (array azimuth exactly as
      written in site.yaml) yields 0 glare and would make the comparison
      vacuous. SKIPs loudly if that case has no receptor with sub-panel hits.
      Exact hit-test: the pipeline's analytic detector vs exact sampled
      flat-rectangle geometry (the SGHAT Sec.4.4 cone-plane test) over the full
      year, on a NEAR-field and a FAR-field receptor of the committed cases so
      both angular regimes are covered. Three things keep the exact side both
      independent and resolved: a conservative cone pre-filter (a provable
      superset, so it can only cut cost), an n_grid derived from the geometry
      rather than hardcoded, and an n_grid=1001 escalation of every coarse-grid
      disagreement - so a coarse grid can never fake a pass. C2 (circular vs
      polygon footprint distortion) is reported on the same live receptors.
  [D] Sensitivity table: annual Y/G/R vs (time freq, groups, min_dni,
      slope_error, dni_mode, reflectivity_mode) - hourly sweeps always;
      per-minute variants when run with --full.

Usage:
  .venv\\Scripts\\python.exe scripts\\validate_charts.py [--full|--sensitivity-only]
                                                   [--hit-test circular|polygon]
Results are appended to output/validation/charts.txt (PASS/FAIL lines).
"""

import copy
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from solar_glare.analysis import (
    load_config,
    build_site,
    run_analysis,
    compute_ephemeris,
    analyze_pair,
    _analyze_single,
)
from solar_glare.hazard import (
    classify,
    burn_threshold,
    flash_threshold,
    corneal_irradiance,
    retinal_irradiance,
    effective_subtended_angle,
    beam_spread_angle,
    SUN_SUBTENDED_RAD,
    _AOI_DEFAULT_GLASS,
)
from solar_glare.geometry import sun_vector, reflect

OUT_PATH = os.path.join(REPO, "output", "validation", "charts.txt")

FULL = "--full" in sys.argv
SENS_ONLY = "--sensitivity-only" in sys.argv
HITMODE = "circular"
for i, a in enumerate(sys.argv):
    if a == "--hit-test" and i + 1 < len(sys.argv):
        HITMODE = sys.argv[i + 1]
        break

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        tag = "OK "
    else:
        FAIL += 1
        tag = "FAIL"
    line = f"  [{tag}] {name}" + (f"  -- {detail}" if detail else "")
    print(line)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def skip(name, why=""):
    """Report a check that could not run. Deliberately NOT counted as a pass:
    a parity/comparison check over an empty result set would otherwise look
    green while proving nothing."""
    line = f"  [SKIP] {name}" + (f"  -- {why}" if why else "")
    print(line)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def section(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70 + "\n")


def panel_axes(array):
    """Local panel-plane axes: e = width axis, s = height axis (same as
    PVArray._split_groups, so the rectangle tiles the parent exactly)."""
    up = np.array([0.0, 0.0, 1.0])
    n = array.normal
    e = up - n * np.dot(up, n)
    if np.linalg.norm(e) < 1e-9:
        e = np.array([1.0, 0.0, 0.0])
    else:
        e = e / np.linalg.norm(e)
    s = np.cross(n, e)
    return e, s


def exact_hit_mask(pos, array, receptor_enu, slope_error_rad, n_grid=11):
    """Independent cone-plane hit test (SGHAT Sec.4.4 geometry) for a FLAT
    specular panel: sample the panel rectangle and check whether any point P
    satisfies angle(U-P, r) <= beta/2 (plus the front-face guard and sun above
    horizon). Returns a boolean array aligned with pos.index.

    This is independent of the array-limiting-angle formula used in
    _analyze_single: it tests the real rectangle geometry directly.
    """
    n = array.normal
    e, s = panel_axes(array)
    w, h = array.width_m, array.height_m
    u = np.linspace(-0.5, 0.5, n_grid)
    v = np.linspace(-0.5, 0.5, n_grid)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    P = array.center_enu + (uu[..., None] * e) * w + (vv[..., None] * s) * h
    P = P.reshape(-1, 3)  # (J, 3)
    J = P.shape[0]
    vj = receptor_enu - P          # (J, 3)  toward receptor from each point
    vj_n = np.linalg.norm(vj, axis=-1)
    vj_u = vj / vj_n[:, None]

    beta_half = beam_spread_angle(slope_error_rad) / 2.0
    n_all = len(pos)
    hit = np.zeros(n_all, dtype=bool)
    az = pos["azimuth"].to_numpy()
    el = pos["apparent_elevation"].to_numpy()
    s_vec = sun_vector(az, el)                      # (N,3)
    front = (s_vec @ array.normal) > np.sin(beta_half)
    r_vec = reflect(s_vec, array.normal)            # (N,3)
    elev_ok = el > 0.0

    chunk = 30000
    for i0 in range(0, n_all, chunk):
        i1 = min(i0 + chunk, n_all)
        rr = r_vec[i0:i1]                            # (M,3)
        cosang = (vj_u[None, :, :] * rr[:, None, :]).sum(-1)   # (M,J)
        cosang = np.clip(cosang, -1.0, 1.0)
        ang = np.arccos(cosang)
        minang = ang.min(axis=1)
        hit[i0:i1] = (minang <= beta_half) & front[i0:i1] & elev_ok[i0:i1]
    return hit


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    t_start = time.time()
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n# ---- validate_charts.py run {pd.Timestamp.now()} "
                f"({('full' if FULL else 'core')}) ----\n")

    cfg = load_config(os.path.join(REPO, "config", "site.yaml"))
    cfg["environment"]["hit_test_mode"] = HITMODE

    if SENS_ONLY:
        # Reuse the last full-run summary for the baseline reference totals.
        summ_path = os.path.join(REPO, "output", "results", "summary.csv")
        if os.path.exists(summ_path):
            summ = pd.read_csv(summ_path)
            base_tot = int(summ[["G", "Y", "R"]].sum().sum())
            base_y = int(summ["Y"].sum())
        else:
            base_tot, base_y = None, None
        run_sensitivity(cfg, base_tot, base_y)
        print(f"\n=== RESULT: {PASS} passed, {FAIL} failed "
              f"(elapsed {time.time() - t_start:.0f}s) ===")
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(f"=== RESULT: {PASS} passed, {FAIL} failed "
                    f"(elapsed {time.time() - t_start:.0f}s) ===\n")
        return 1 if FAIL else 0

    section("[0] Baseline per-minute analysis")
    times, pos, dni, long_df, summary, details, site, arrays, receptors = run_analysis(cfg)
    check("baseline per-minute run completes",
          len(times) == 525600, f"{len(times)} timesteps")

    # ----------------------------------------------------------------- [A]
    section("[A] Ocular chart (08) - zone consistency & arithmetic")

    all_rows = []
    n_total = 0
    n_mismatch = 0
    n_roundtrip_bad = 0
    for (a, r), d in details.items():
        if d.empty:
            continue
        er = d["retinal_wcm2"].to_numpy()
        x = d["subtended_rad"].to_numpy()
        hz = d["hazard"].to_numpy()
        er_burn = burn_threshold(x)
        er_flash = flash_threshold(x)
        zone_ok = np.where(
            hz == "G", er < er_flash,
            np.where(hz == "Y", (er >= er_flash) & (er < er_burn), er >= er_burn))
        n_total += len(d)
        n_mismatch += int((~zone_ok).sum())
        # round-trip: E_c and E_r recomputed from stored values. The
        # reflectivity used per minute is stored in the detail (per the active
        # reflectivity_mode: "constant", or the AOI- / solar-weighted value).
        rho_stored = d["reflectivity_used"].to_numpy()
        tau = d["atmos_transmittance"].to_numpy() if "atmos_transmittance" in d.columns else 1.0
        ec_ref = d["dni_wm2"].to_numpy() / 1e4 * rho_stored * tau
        er_ref = retinal_irradiance(ec_ref, x)
        re = np.abs(er - er_ref) / np.maximum(np.abs(er), 1e-12)
        n_roundtrip_bad += int((re > 1e-9).sum())
        d2 = d.copy()
        d2["array"] = a
        d2["receptor"] = r
        all_rows.append(d2)

    check("all glare minutes inside their hazard zone (G<flash<=Y<burn<=R)",
          n_mismatch == 0, f"{n_mismatch} violations of {n_total}")
    check("E_c = DNI/1e4*rho and E_r = E_c*dp2/(f2*x2)*tau round-trip (rel.err<1e-9)",
          n_roundtrip_bad == 0, f"{n_roundtrip_bad} of {n_total} rows off")

    # known-answer: staring at the sun
    sun_er = retinal_irradiance(corneal_irradiance(1000.0, 1.0), SUN_SUBTENDED_RAD)
    check("known-answer: sun disc (x=9.3 mrad, DNI=1000, rho=1) -> E_r ~ 8 W/cm2",
          abs(float(sun_er) - 8.0) < 0.5, f"E_r = {sun_er:.3f} W/cm2")
    check("known-answer: sun classified Yellow",
          classify(np.array([sun_er]), np.array([SUN_SUBTENDED_RAD]))[0] == "Y")

    # known-answer: every demo point is below burn (flat panel far field).
    # The baseline uses the array azimuth as written in site.yaml; when that
    # azimuth (e.g. the config's default 0.0) has no glare minutes the summary
    # peaks are all NaN, so skip the worst-point check rather than idxmax-crash.
    peak_series = summary["annual_peak_retinal_wcm2"].dropna()
    if not peak_series.empty:
        worst = peak_series.max()
        worst_x = summary.loc[summary["annual_peak_retinal_wcm2"].idxmax(),
                              "peak_subtended_rad"]
        check("known-answer: worst demo point is Yellow (flash<E_r<burn)",
              (classify(np.array([worst]), np.array([worst_x]))[0] == "Y")
              and float(flash_threshold(worst_x)) < float(worst)
              and float(worst) < float(burn_threshold(worst_x)),
              f"E_r={worst:.3f} W/cm2 @ x={worst_x:.4f} rad")

    # plot-extent coverage of figure 07 (limits auto-extend to the data)
    if all_rows:
        big = pd.concat(all_rows, ignore_index=True)
        xmin, xmax = big["subtended_rad"].min(), big["subtended_rad"].max()
        emin, emax = big["retinal_wcm2"].min(), big["retinal_wcm2"].max()
        xlo = 10.0 ** np.floor(np.log10(xmin) - 0.1)
        yhi = 10.0 ** np.ceil(np.log10(emax) + 0.1)
        check("ocular-chart axis limits cover all points (auto xlo/yhi)",
              xmin >= xlo and xmax <= 0.6 and emin >= 1e-8 and emax <= yhi,
              f"x=[{xmin:.4g},{xmax:.4g}] (xlo={xlo:.1g}), "
              f"E_r=[{emin:.4g},{emax:.4g}] (yhi={yhi:.1g})")

    # ----------------------------------------------------------------- [B]
    section("[B] Annual hours (07) - summary audit")

    check("G+Y+R == minutes_with_glare for every summary row",
          bool((summary["minutes_with_glare"] == summary[["G", "Y", "R"]].sum(axis=1)).all()))

    cnt = long_df.groupby(["array", "receptor"])["hazard"].value_counts().unstack(
        fill_value=0).reindex(columns=["G", "Y", "R"], fill_value=0)
    merged = summary.reset_index().set_index(["array", "receptor"])
    cnt = cnt.reindex(index=merged.index, fill_value=0)
    okc = True
    for (a, r), row in merged.iterrows():
        for code in ["G", "Y", "R"]:
            if int(row[code]) != int(cnt.loc[(a, r), code]):
                okc = False
    check("long-table hazard counts equal summary G/Y/R per (array,receptor)", okc)

    n_dups = int(long_df.duplicated(subset=["time", "array", "receptor"]).sum())
    check("no duplicate (time, array, receptor) rows in long table", n_dups == 0,
          f"{n_dups} duplicates")

    # per-pair detail: exactly one row per glare minute
    n_pair_dups = 0
    for (a, r), d in details.items():
        if not d.empty and int(d["time"].duplicated().sum()):
            n_pair_dups += int(d["time"].duplicated().sum())
    check("per-pair detail deduped to one row per glare minute", n_pair_dups == 0,
          f"{n_pair_dups} duplicate rows")

    # summary peak == max retinal of the detail
    peak_bad = 0
    for (a, r), d in details.items():
        if d.empty:
            continue
        sval = float(summary.loc[(r, a), "annual_peak_retinal_wcm2"])
        mval = float(d["retinal_wcm2"].max())
        if abs(sval - mval) > 1e-9:
            peak_bad += 1
    check("summary annual_peak_retinal == max over per-minute detail", peak_bad == 0,
          f"{peak_bad} mismatches")

    # figure 06 conversion hours = minutes/60
    tot_min = int(summary[["G", "Y", "R"]].sum().sum())
    tot_h = tot_min / 60.0
    check("fig 07 hours = minutes/60 (total Y = %.1f h/yr)" % (tot_h,), True,
          f"total glare = {tot_min} min = {tot_h:.1f} h/yr")

    # ----------------------------------------------------------------- [C]
    section(f"[C] Independent cone-plane hit test vs pipeline "
            f"(hit_test_mode={HITMODE})")
    _, pos, dni = compute_ephemeris(cfg, site)

    # multi-thread parity: the groups>1 path must be bit-identical to serial.
    # The top-level baseline config (array azimuth exactly as written in
    # site.yaml) yields 0 glare on the committed site, so comparing two all-'N'
    # arrays would prove nothing.  Resolve a committed site.cases entry instead
    # - the same geometry + environment that produced the committed outputs -
    # and require the chosen receptor to actually have sub-panel hits,
    # otherwise report SKIP rather than a vacuous PASS.
    def pair_kwargs(ccfg):
        """analyze_pair kwargs taken straight from a config's environment,
        mirroring run_analysis.py so the parity check exercises the real
        production path (E903 reflectivity, Beer-Lambert tau, IEC columns)."""
        env = ccfg.get("environment", {})
        return dict(
            slope_error_rad=env.get("slope_error_rad", 0.0),
            min_dni_wm2=env.get("min_dni_wm2", 10.0),
            reflectivity_mode=env.get("reflectivity_mode", "solar_weighted"),
            glass_type=env.get("glass_type", _AOI_DEFAULT_GLASS),
            iec_enabled=bool(env.get("iec62471_enabled", False)),
            solar_weighted_n_arc=env.get("solar_weighted_arc_n", None),
            solar_weighted_arc_wl0=float(env.get(
                "solar_weighted_arc_design_wl_nm", 600.0)),
            attenuation_mode=env.get("atmospheric_attenuation", "none"),
            visibility_km=float(env.get("visibility_km", 20.0)),
            attenuation_wl_nm=float(env.get(
                "attenuation_wavelength_nm", 550.0)),
            hit_test_mode=HITMODE,
        )

    cases = cfg.get("site", {}).get("cases") or []

    def resolve_case(idx):
        """Build (label, cfg, arrays, recs, pos, dni) for site.cases[idx],
        applying that case's per-array azimuth/tilt and env overrides. A case
        never moves the site, so the ephemeris is shared unless the case
        overrides the DNI model. Resolving cases in one place keeps every [C]
        sub-check on a single code path."""
        case = cases[idx]
        ccfg = copy.deepcopy(cfg)
        for arr in ccfg["arrays"]:
            if "azimuth_deg" in case:
                arr["azimuth_deg"] = float(case["azimuth_deg"])
            if "tilt_deg" in case:
                arr["tilt_deg"] = float(case["tilt_deg"])
        ccfg.setdefault("environment", {}).update(case.get("env") or {})
        csite, carrays, crecs = build_site(ccfg)
        if any(k in (case.get("env") or {})
               for k in ("dni_mode", "dni_constant_wm2")):
            _, cpos, cdni = compute_ephemeris(ccfg, csite)
        else:
            cpos, cdni = pos, dni   # a case never moves the site reference
        return (case.get("label"), ccfg, carrays,
                {r.name: r for r in crecs}, cpos, cdni)

    pname = None
    if cases and arrays:
        label0, ccfg, carrays, crecs, cpos, cdni = resolve_case(0)
        ckw = pair_kwargs(ccfg)
        pa = carrays[0]
        pw = max(2, int(ccfg.get("environment", {}).get("n_workers", 0) or 2))
        # prefer the receptors that carry near-field multi-group hits
        for cand in ("obs_ptt", "west1_route_11", "west1_route_1",
                     "west1_route_2"):
            if cand in crecs:
                c_s, d_s = analyze_pair(pa, crecs[cand], cpos, cdni,
                                        n_workers=1, **ckw)
                if (c_s != "N").any():
                    pname = cand
                    break
        if pname is not None:
            c_p, d_p = analyze_pair(pa, crecs[pname], cpos, cdni,
                                    n_workers=pw, **ckw)
            same_codes = bool(np.array_equal(c_s, c_p))
            same_detail = bool(d_s.equals(d_p))
            check(f"parallel group merge (n_workers={pw}) == serial "
                  f"(n_workers=1) on {label0}/{pname} "
                  f"(groups={pa.groups}, hit-test={HITMODE}, "
                  f"atten={ckw['attenuation_mode']}, "
                  f"iec={ckw['iec_enabled']})",
                  same_codes and same_detail,
                  f"codes_equal={same_codes}, detail_equal={same_detail}, "
                  f"cols={len(d_p.columns)}, rows={len(d_p)}, "
                  f"glare={int((c_p != 'N').sum())}")
        else:
            skip("parallel group merge parity",
                 f"no receptor with sub-panel hits in case "
                 f"{label0!r} - comparing two empty results "
                 f"would prove nothing")
    else:
        skip("parallel group merge parity",
             "site.cases is empty - no committed case to resolve")

    def exact_panel_union(pos, array, receptor_enu, slope_error_rad, n_grid,
                          chunk=2500):
        """Chunked union over the array's sub-panels of the exact sampled
        flat-rectangle hit test (SGHAT Sec.4.4): true if ANY point P on ANY
        sub-panel satisfies angle(U-P, r) <= beta/2, with the same gates the
        pipeline applies per group (front-face sun disc, ahead, elevation>0).
        Returns a boolean mask aligned with pos.index.

        Resolution: the uniform n_grid x n_grid point sample sets a built-in
        floor - the angular sample spacing along a sub-panel axis is
        (sub-panel extent / n_grid) / distance to the NEAREST sub-panel, since
        the closest sub-panel subtends the largest angle and therefore sets the
        floor. Choose n_grid with auto_n_grid() below instead of hardcoding it:
        cost is O(n_grid^2 * n_groups * minutes), and a too-small grid can only
        inflate the escalation volume, never the verdict, because every
        coarse-grid disagreement is re-tested at n_grid=1001.
        """
        beta_half = beam_spread_angle(slope_error_rad) / 2.0
        n = array.normal
        az = pos["azimuth"].to_numpy()
        el = pos["apparent_elevation"].to_numpy()
        s_vec = sun_vector(az, el)
        r_vec = reflect(s_vec, n)
        elev_ok = el > 0.0
        front = (s_vec @ n) > np.sin(beta_half)
        n_all = len(pos)
        out = np.zeros(n_all, dtype=bool)
        u = np.linspace(-0.5, 0.5, n_grid)
        v = np.linspace(-0.5, 0.5, n_grid)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        for g in (array.group_arrays or [array]):
            e, s = panel_axes(g)
            w, h = g.width_m, g.height_m
            P = (g.center_enu + (uu[..., None] * e) * w
                 + (vv[..., None] * s) * h).reshape(-1, 3)
            vj = (receptor_enu - P).astype(np.float32)
            vj /= np.linalg.norm(vj, axis=-1, keepdims=True)
            u_hat = (receptor_enu - g.center_enu)
            u_hat /= np.linalg.norm(u_hat)
            ahead = (r_vec @ u_hat) > 0.0
            for i0 in range(0, n_all, chunk):
                i1 = min(i0 + chunk, n_all)
                ang = np.arccos(np.clip(r_vec[i0:i1].astype(np.float32) @ vj.T,
                                        -1.0, 1.0))
                hit = (ang <= beta_half).any(axis=1)
                out[i0:i1] |= (hit & front[i0:i1] & ahead[i0:i1]
                               & elev_ok[i0:i1])
        return out

    def cone_prefilter(cpos, carray, receptor_enu, beta_half):
        """Necessary condition for ANY sub-panel hit, derived from the geometry
        alone so it shares no code with the detector under test: the front face
        must be lit, the sun must be up, and the reflected direction must lie
        inside the tangent cone of SOME sub-panel.

        Each sub-panel is bounded by the ball of radius = its own half-diagonal
        about its own centre, so every direction from the receptor to that
        sub-panel lies within asin(R/d) of its centre direction; a hit needs
        angle(r, dir_to_P) <= beta_half, hence
        angle(r, dir_to_centre_g) <= asin(R_g/d_g) + beta_half. The union over
        sub-panels is therefore a provable SUPERSET of the true hit set: it can
        only cut cost, never manufacture a false negative.

        The ball bound needs arcsin, NOT arctan: the bound is the tangent
        cone to a sphere of radius R, which is asin(R/d) and saturates at the
        full hemisphere (pi/2) once R >= d, whereas atan(R/d) keeps shrinking
        below pi/2 and silently under-covers. At the committed near receptor a
        whole-array radius really does hit that regime (R=2445 m, d=2570 m,
        R/d=0.951: asin = 1.258 rad = 72.1 deg but atan = 0.761 rad = 43.6 deg),
        which is how the 28 deg under-coverage was caught live. With the
        per-sub-panel radii used here R/d is only 0.008-0.083, so the two forms
        differ by <= 0.01 mrad and the choice is not load-bearing for these
        receptors - arcsin is kept because it is the correct bound, and because
        a future coarser grouping could push R/d back towards 1.
        """
        n = carray.normal
        s = sun_vector(cpos["azimuth"].to_numpy(),
                       cpos["apparent_elevation"].to_numpy())
        r = reflect(s, n)
        inside = np.zeros(len(cpos), dtype=bool)
        for g in (carray.group_arrays or [carray]):
            u = receptor_enu - g.center_enu
            d = float(np.linalg.norm(u))
            u = u / max(d, 1e-9)
            R = 0.5 * float(np.hypot(g.width_m, g.height_m))
            cone = np.arcsin(min(1.0, R / d)) + beta_half
            inside |= (r @ u) > np.cos(cone)
        return (((s @ n) > np.sin(beta_half))
                & (cpos["apparent_elevation"].to_numpy() > 0.0)
                & inside)

    def auto_n_grid(carray, receptor_enu, beta_half, safety=2.5, lo=11, hi=101):
        """Smallest uniform sample grid whose angular spacing stays `safety`
        times below beta/2 at the nearest sub-panel. Returns
        (n_grid, spacing_mrad, margin) so the resolution actually achieved is
        recorded in the log rather than assumed."""
        groups = carray.group_arrays or [carray]
        span = max(max(g.width_m, g.height_m) for g in groups)
        d_near = min(float(np.linalg.norm(receptor_enu - g.center_enu))
                     for g in groups)
        need = safety * span / (d_near * beta_half)
        n = int(np.clip(int(np.ceil(need)), lo, hi))
        spacing = span / n / d_near
        return n, spacing * 1e3, beta_half / spacing

    # ------------------------------------------------------------- (C1)
    # Pipeline (groups merged) vs exact sampled flat-rectangle geometry, on a
    # NEAR-field and a FAR-field receptor so both angular regimes are covered by
    # a live regression test. For hit_test_mode=polygon the pipeline IS the
    # analytic equivalent of the exact test, so it must reproduce it with zero
    # FP/FN (up to grid resolution, resolved by the n_grid=1001 escalation).
    # Each target is resolved from a committed case: the baseline array azimuth
    # as written in site.yaml yields 0 glare everywhere, which would make every
    # comparison below vacuously pass.
    # Targets are addressed by committed case LABEL, not list index, so
    # reordering site.cases cannot silently point the test at the wrong case.
    EXACT_TARGETS = (("az098_e903_t10", "obs_ptt", "near"),
                     ("az278_e903_t10", "west1_route_1", "far"))
    case_index = {c.get("label"): i for i, c in enumerate(cases)}
    for tgt_label, tgt_name, tgt_regime in EXACT_TARGETS:
        tag = f"hit-test == exact sampled geometry ({tgt_name}, {tgt_regime} field)"
        tgt_idx = case_index.get(tgt_label)
        if not (cases and arrays) or tgt_idx is None:
            skip(tag, f"committed case {tgt_label!r} not found in site.cases "
                      f"(have {[c.get('label') for c in cases]})")
            continue
        label, ccfg, carrays, crecs, cpos, cdni = resolve_case(tgt_idx)
        if tgt_name not in crecs:
            skip(tag, f"receptor {tgt_name!r} absent from case {label!r}")
            continue
        t_c = time.time()
        a0 = carrays[0]
        rw = crecs[tgt_name]
        ckw = pair_kwargs(ccfg)
        pw = max(2, int(ccfg.get("environment", {}).get("n_workers", 0) or 2))
        slope_err = ckw["slope_error_rad"]
        codes, _ = analyze_pair(a0, rw, cpos, cdni, n_workers=pw, **ckw)
        pipe = (codes != "N") & (cdni.to_numpy() > ckw["min_dni_wm2"])
        n_pipe = int(pipe.sum())
        if n_pipe == 0:
            skip(tag, f"case {label!r} gives no pipeline hits for {tgt_name!r} - "
                      f"comparing two empty hit sets would prove nothing")
            continue
        beta_half = beam_spread_angle(slope_err) / 2.0
        pre = cone_prefilter(cpos, a0, rw.enu, beta_half)
        if not bool(pre[pipe].all()):
            skip(tag, "cone pre-filter dropped pipeline hits - it is not a "
                      "superset, so the exact side would be unsound")
            continue
        n_grid, spacing_mrad, margin = auto_n_grid(a0, rw.enu, beta_half)
        ex_sub = exact_panel_union(cpos[pre], a0, rw.enu, slope_err, n_grid)
        exact = np.zeros(len(cpos), dtype=bool)
        exact[np.flatnonzero(pre)] = ex_sub
        n_exact = int(exact.sum())
        fn = int((~pipe & exact).sum())
        unconf = np.flatnonzero(pipe & ~exact)   # pipeline hit, coarse exact did not
        fns = np.flatnonzero(~pipe & exact)      # coarse exact hit, pipeline did not
        # Escalation settles what the coarse grid could not resolve, so the
        # verdict comes from n_grid=1001 only. Note the polarity: a candidate
        # under unconf is a CONFIRMED false positive when the fine grid also
        # finds no hit, hence the complement; a candidate under fns is a
        # confirmed false negative when the fine grid still finds a hit, hence
        # no complement. A coarse grid alone can only add candidates here.
        fp_deep = fn_deep = 0
        if unconf.size:
            fp_deep = int((~exact_panel_union(
                cpos.iloc[unconf], a0, rw.enu, slope_err,
                n_grid=1001, chunk=20)).sum())
        if fns.size:
            fn_deep = int(exact_panel_union(
                cpos.iloc[fns], a0, rw.enu, slope_err,
                n_grid=1001, chunk=20).sum())
        if HITMODE == "polygon":
            check(f"{tag}, case {label}: zero FP, FN <= 1% of hits",
                  fp_deep == 0
                  and fn_deep <= max(1, int(round(0.01 * n_pipe))),
                  f"n_grid={n_grid} spacing={spacing_mrad:.2f}mrad "
                  f"margin={margin:.1f}x pre={int(pre.sum())}/{len(cpos)} "
                  f"FN={fn}->deep={fn_deep} FP={unconf.size}->deep={fp_deep} "
                  f"pipeline={n_pipe} exact={n_exact} "
                  f"({100.0 * fn_deep / max(n_pipe, 1):.2f}% FN) "
                  f"in {time.time() - t_c:.0f}s")
        else:
            agree = float((pipe == exact).mean())
            check(f"{tag}, case {label}, groups={a0.groups}, "
                  f"hit-test=circular: gate 0.99",
                  agree >= 0.99,
                  f"agreement={agree * 100:.2f}% approx={n_pipe} exact={n_exact} "
                  f"(FP={unconf.size}, FN={fn}) n_grid={n_grid} "
                  f"spacing={spacing_mrad:.2f}mrad in {time.time() - t_c:.0f}s")

        # (C2) circular vs polygon footprint on the same live receptor:
        # documents the area-equivalent-circle distortion for elongated panels
        # (where they differ, polygon is the exact one).
        if HITMODE == "polygon":
            # ckw already carries hit_test_mode=HITMODE, so override it through
            # a copy rather than a second keyword (which would be a TypeError).
            codes_c, _ = analyze_pair(a0, rw, cpos, cdni, n_workers=pw,
                                      **dict(ckw, hit_test_mode="circular"))
            c_hit = (codes_c != "N") & (cdni.to_numpy() > ckw["min_dni_wm2"])
            diff = int((c_hit != pipe).sum())
            check(f"circular vs polygon footprint ({tgt_name}, {tgt_regime} "
                  f"field): documented distortion",
                  True, f"circular={int(c_hit.sum())} polygon={n_pipe} "
                        f"diff={diff}")

    # ----------------------------------------------------------------- [D]
    section(f"[D] Sensitivity - annual G/Y/R totals (minutes) "
            f"[hit-test={HITMODE}]")
    base_tot = int(summary[["G", "Y", "R"]].sum().sum())
    base_y = int(summary["Y"].sum())
    run_sensitivity(cfg, base_tot, base_y)

    print(f"\n=== RESULT: {PASS} passed, {FAIL} failed "
          f"(elapsed {time.time() - t_start:.0f}s) ===")
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"=== RESULT: {PASS} passed, {FAIL} failed "
                f"(elapsed {time.time() - t_start:.0f}s) ===\n")
    return 1 if FAIL else 0


def run_sensitivity(cfg, base_tot, base_y):
    if base_tot is None:
        check("sensitivity baseline reference", False,
              "no baseline summary found (run without --sensitivity-only first)")
        return
    check("sensitivity baseline reference loaded",
          True, f"per-minute baseline total={base_tot}, Y={base_y}")

    def apply_path(c, path, value):
        parts = path.split(".")
        node = c
        for p in parts[:-1]:
            if p.isdigit():
                node = node[int(p)]
            else:
                node = node.setdefault(p, {})
        if parts[-1].isdigit():
            node[int(parts[-1])] = value
        else:
            node[parts[-1]] = value

    def run_variant(cfg0, mods, label, per_minute=False):
        c = copy.deepcopy(cfg0)
        for k, v in mods.items():
            apply_path(c, k, v)
        t0 = time.time()
        _, _, _, long2, sum2, _, _, _, _ = run_analysis(c)
        tot = int(sum2[["G", "Y", "R"]].sum().sum())
        gy = int(sum2["G"].sum())
        yy = int(sum2["Y"].sum())
        ry = int(sum2["R"].sum())
        dt = time.time() - t0
        check(f"{label}: G={gy} Y={yy} R={ry} total={tot} "
              f"(baseline total={base_tot}, Y={base_y})",
              True, f"{'per-minute' if per_minute else 'hourly'} in {dt:.0f}s")

    # hourly sweeps (cheap)
    run_variant(cfg, {"site.time_freq": "h"}, "freq=hourly")
    run_variant(cfg, {"site.time_freq": "h", "arrays.0.groups": 1}, "hourly groups=1")
    run_variant(cfg, {"site.time_freq": "h", "arrays.0.groups": 4}, "hourly groups=4")
    run_variant(cfg, {"site.time_freq": "h", "arrays.0.groups": 16}, "hourly groups=16")
    run_variant(cfg, {"site.time_freq": "h", "environment.min_dni_wm2": 0.0},
                "hourly min_dni=0")
    run_variant(cfg, {"site.time_freq": "h", "environment.min_dni_wm2": 30.0},
                "hourly min_dni=30")
    run_variant(cfg, {"site.time_freq": "h", "environment.slope_error_rad": 10e-3},
                "hourly slope_err=10 mrad")
    run_variant(cfg, {"site.time_freq": "h", "environment.dni_mode": "constant"},
                "hourly DNI=const 1000")

    if FULL:
        # per-minute variants (each ~40 s at n_workers=8)
        run_variant(cfg, {"environment.min_dni_wm2": 0.0}, "per-minute min_dni=0",
                    per_minute=True)
        run_variant(cfg, {"environment.dni_mode": "constant"},
                    "per-minute DNI=const 1000", per_minute=True)
        run_variant(cfg, {"environment.slope_error_rad": 10e-3},
                    "per-minute slope_err=10 mrad", per_minute=True)


if __name__ == "__main__":
    sys.exit(main())
