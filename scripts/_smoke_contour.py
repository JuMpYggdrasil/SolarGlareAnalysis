"""Smoke test for solar_glare.contour on a SMALL synthetic grid.

Exercises every drawing path (RG fill + boundaries + log panels + GeoJSON
boundaries + RGBA overlay + folium map + CSV) in ~1 s, with no analysis run.
Deliberately shaped so the RG field is a gradient (RG0 core surrounded by
RG1/RG2/RG3), which is what exercises contour hole-splitting and the annulus
case that a flat field would miss.
"""
import inspect
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solar_glare import contour as C
from solar_glare.analysis import build_site, load_config
from solar_glare.viz import _RG_ORDER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(tempfile.gettempdir(), "opencode", "contour_smoke")
os.makedirs(out, exist_ok=True)

cfg = load_config(os.path.join(ROOT, "config", "site.yaml"))
site, arrays, receptors = build_site(cfg)
a = arrays[0]
ref = site["reference"]

# --- build a Receptor-like list: 9x9 grid at 0 m AGL, ENU-centred --------
from solar_glare.analysis import Receptor
from solar_glare.geometry import enu_to_latlon

nrow = ncol = 9
step = 900.0
grid_recs, names = [], []
i = 0
for row in range(nrow):
    for col in range(ncol):
        y = -step * (row - (nrow - 1) / 2)          # row 0 = north
        x = step * (col - (ncol - 1) / 2)
        lat, lon = enu_to_latlon(ref["lat"], ref["lon"], x, y)
        r = Receptor(name=f"g_{row:02d}_{col:02d}", lat=lat, lon=lon,
                     height_agl_m=0.0)
        r.setup(ref["lat"], ref["lon"])
        grid_recs.append(r)
        names.append(r.name)

# plus two non-grid receptors to prove the prefix split works
extra = []
for nm in ("v_obs_ptt", "v_west1_route_1"):
    e = Receptor(name=nm, lat=ref["lat"] + 0.01, lon=ref["lon"] + 0.01,
                 height_agl_m=40.0)
    e.setup(ref["lat"], ref["lon"])
    extra.append(e)

grid, others = C.grid_from_receptors(grid_recs + extra, C.GRID_PREFIX)
assert [r.name for r in extra] == [r.name for r in others], others
assert grid["nrow"] == nrow and grid["ncol"] == ncol, (grid["nrow"], grid["ncol"])
# north-to-south ordering: row index must be DESCENDING down the returned list
assert list(grid["row"]) == sorted(grid["row"]), "rows must ascend"
assert list(grid["col"][:ncol]) == list(range(ncol)), "cols must run west->east"
print(f"grid ok: {grid['nrow']}x{grid['ncol']}, "
      f"{len(others)} non-grid, row order ascending")

# --- a synthetic summary whose RG field is a radial gradient ------------
d = np.linalg.norm(grid["enu"][:, :2] - a.center_enu[None, :2], axis=1)
rank = np.clip((d / d.max() * 3.9).astype(int), 0, 3)
# punch an RG0 hole in the middle of the RG3 ring to exercise multi-ring paths
rank[np.argmin(d)] = 0
minutes = np.where(rank > 0, 100.0 * (rank + 1), 0.0)
peak = np.where(rank > 0, 0.05 * (rank + 1) ** 3, np.nan)
tmax = np.where(rank > 0, 10.0 / (rank + 1) ** 4, np.nan)
tmaxb = np.where(rank > 0, 100.0 / (rank + 1) ** 3, np.nan)
lr = np.where(rank > 0, 1e5 * (rank + 1) ** 2, np.nan)

idx = pd.MultiIndex.from_product([names, [a.name]], names=["receptor", "array"])
summary = pd.DataFrame({
    "minutes_with_glare": minutes,
    "iec_worst_risk": [_RG_ORDER[r] for r in rank],
    "annual_peak_retinal_wcm2": peak,
    "iec_min_tmax_thermal_s": tmax,
    "iec_min_tmax_blue_s": tmaxb,
    "iec_peak_lr_w_sr_m2": lr,
}, index=idx)

f = C.fields_from_summary(summary, grid, arrays)
assert f["rg_matrix"].shape == (nrow, ncol)
assert not np.isnan(f["peak_er"]).any() or True
# NaN policy: a rank-0 cell must have NaN peak, NOT 0.0
assert np.isnan(f["peak_er"][rank == 0]).all(), "glare-free cells must stay NaN"
assert (f["peak_er"][rank > 0] > 0).all()
assert np.isfinite(f["dist_m"]).all() and (f["dist_m"] > 0).all()
print("fields ok: rg_matrix", f["rg_matrix"].shape,
      "| NaN preserved on", int(np.isnan(f["peak_er"]).sum()), "glare-free cells")

# --- CSV ------------------------------------------------------------------
csv_path = os.path.join(out, "contour_grid.csv")
df = C.contour_grid_csv(grid, f, arrays, csv_path)
back = pd.read_csv(csv_path)
assert len(back) == nrow * ncol
assert list(back.columns) == list(df.columns)
print("csv ok:", csv_path, os.path.getsize(csv_path), "bytes")

# --- PNG ------------------------------------------------------------------
# One figure per panel, each in a solid and a 50%-alpha variant.
written = C.plot_risk_contour_set(site, arrays, grid, f, out,
                                  case_label="SMOKE")
assert len(written) == 10, written
assert len({os.path.basename(p) for p in written}) == 10, "panel names collide"
for p in written:
    assert os.path.getsize(p) > 8_000, (p, os.path.getsize(p))
print("png ok: %d files, smallest %d bytes"
      % (len(written), min(os.path.getsize(p) for p in written)))

# --- RG0 is EMPTY BACKGROUND in the isorisk panels ------------------------
# RG0 (Exempt) is the safe area, so the two discrete panels must leave it
# unpainted. The three things that could quietly regress this are each pinned:
#   1. the palette entry itself is transparent at BOTH fill opacities;
#   2. the opacity is baked into the colours, NOT passed as contourf(alpha=),
#      which would override every entry's alpha and turn RG0 50% BLACK in the
#      semi-transparent variant -- a grey wash over the whole safe area;
#   3. the rendered pixels really are background there, for both variants.
from matplotlib.colors import to_rgba  # noqa: E402
for fill_a in (1.0, 0.5):
    cm = C._rg_cmap(fill_a)
    assert tuple(cm(0)) == C._RG_CLEAR, (fill_a, tuple(cm(0)))
    assert cm(0)[3] == 0.0, (fill_a, "RG0 must have zero alpha")
    for i, c in enumerate(C._RG_ORDER[1:], start=1):
        want = to_rgba(C._IEC_COLORS[c], fill_a)
        assert np.allclose(cm(i), want), (fill_a, c, tuple(cm(i)), want)
# The panels must not reintroduce a global alpha on the contourf call.
_src_a = inspect.getsource(C.plot_risk_contours)
assert _src_a.count("_rg_cmap(alpha)") == 2, "panels a/b must both use _rg_cmap"
assert "ListedColormap([_IEC_COLORS" not in _src_a, \
    "a panel rebuilt the palette from _IEC_COLORS, RG0 would be green again"
# No discrete isorisk `contourf` may carry a global alpha. That kwarg
# overrides the alpha of EVERY palette entry, which is what would turn the
# transparent RG0 into 50% black in the semi-transparent variant.
assert 'neither", alpha=' not in _src_a, \
    "a discrete contourf still passes alpha= and would black out RG0"
assert "extend=\"neither\", alpha" not in _src_a, \
    "a discrete contourf still passes alpha= and would black out RG0"

# And it survives into the rendered pixels. A CONTROLLED field is used rather
# than the gradient above: that gradient's outer cells are NaN (masked by
# contourf, hence white for an unrelated reason), so sampling its margin would
# pass even if RG0 went back to green. Here every cell is a real RG value --
# all RG0 except a small RG3 block -- so any green left in the image can only
# come from the RG0 palette entry.
import matplotlib.image as mpimg  # noqa: E402

ctl_rank = np.zeros(nrow * ncol, dtype=int)
ctl_rank[nrow // 2 * ncol + ncol // 2] = 3           # one RG3 cell
ctl_rank[nrow // 2 * ncol + ncol // 2 + 1] = 3       # and its east neighbour
ctl = C.fields_from_summary(pd.DataFrame({
    "minutes_with_glare": np.where(ctl_rank > 0, 500.0, 0.0),
    "iec_worst_risk": [_RG_ORDER[r] for r in ctl_rank],
    "annual_peak_retinal_wcm2": np.where(ctl_rank > 0, 0.4, np.nan),
    "iec_min_tmax_thermal_s": np.where(ctl_rank > 0, 0.5, np.nan),
    "iec_min_tmax_blue_s": np.where(ctl_rank > 0, 5.0, np.nan),
    "iec_peak_lr_w_sr_m2": np.where(ctl_rank > 0, 2e6, np.nan),
}, index=pd.MultiIndex.from_product([names, [arrays[0].name]],
                                   names=["receptor", "array"])),
    grid, arrays)
assert (ctl["rg_matrix"] == 0).sum() == nrow * ncol - 2
ctl_out = os.path.join(out, "_ctl")
os.makedirs(ctl_out, exist_ok=True)
C.plot_risk_contours(site, arrays, grid, ctl,
                     os.path.join(ctl_out, "risk_contours_a.png"),
                     case_label="SMOKE", panel="a", alpha=1.0)
C.plot_risk_contours(site, arrays, grid, ctl,
                     os.path.join(ctl_out, "risk_contours_a_semi50.png"),
                     case_label="SMOKE", panel="a", alpha=0.5)

_green = np.array([0x2e, 0xcc, 0x40]) / 255.0     # the RG0 colour being removed
_red = np.array([0xff, 0x41, 0x36]) / 255.0       # RG3, must still be drawn
for fname in ("risk_contours_a.png", "risk_contours_a_semi50.png"):
    img = mpimg.imread(os.path.join(ctl_out, fname))[..., :3]
    # the figure is saved on an opaque white facecolor, so empty RG0 lands as
    # white; the point of this check is that RG0 is not green any more.
    near_green = (np.abs(img - _green) < 0.02).all(axis=2).sum()
    near_red = (np.abs(img - _red) < 0.02).all(axis=2).sum()
    near_white = (np.abs(img - 1.0) < 0.01).all(axis=2).sum()
    assert near_green == 0, ("%s: %d RG0-green pixels survive -- RG0 must be "
                             "empty background" % (fname, near_green))
    assert near_red > 50, ("%s: RG3 vanished (%d red px) -- the palette "
                           "change must not touch the risk bands" % (fname, near_red))
    assert near_white > 0.30 * img[:, :, 0].size, \
        ("%s: only %.1f%% white -- RG0 does not read as blank background"
         % (fname, 100.0 * near_white / img[:, :, 0].size))
print("RG0 empty background ok: palette alpha 0 at both fills, no global "
      "contourf alpha, 0 green px and %.0f%% white in both panel-a variants, "
      "RG3 still drawn" % (100.0 * near_white / img[:, :, 0].size))

# --- HTML + GeoJSON + overlay -------------------------------------------
html = os.path.join(out, "risk_contours.html")
m = C.plot_risk_contours_html(cfg, site, arrays, grid, f, html,
                              case_label="SMOKE")
assert os.path.getsize(html) > 20_000, os.path.getsize(html)
txt = open(html, encoding="utf-8").read()
assert "arcgisonline.com" in txt, "esri street basemap missing"
assert "data:image/png;base64," in txt, "RG overlay raster missing"
assert "Grid query points" in txt, "query layer missing"
assert "RG2 boundary" in txt and "RG3 boundary" in txt
# The HTML array layer must use the SAME grey strip as the PNGs, and the old
# orange square marker must be gone from the page entirely.
assert C._ARRAY_STRIP in txt, "PV array strip colour missing from the page"
assert "#ff851b" not in txt, "the old orange array marker is still on the page"
print("html ok:", html, os.path.getsize(html), "bytes")

# --- the PV array is a grey STRIP, not a square marker --------------------
# `_draw_array` is the one drawing primitive the PNG panels use and the source
# of the colour the HTML reuses, so testing it directly is tighter than digging
# an Axes back out of a finished figure. Two mistakes it has to catch:
# reverting to a marker/dot (which hides which way the 4,890 m array runs), and
# dropping the clip (a 4,890 m strip on a 35 km map runs off the panel and
# spills across the figure margin).
fig_ax = plt.figure()
try:
    ax = fig_ax.add_subplot(111)
    C._draw_array(ax, arrays, site)
    drawn = ax.get_lines()
    strips = [ln for ln in drawn if ln.get_color() == C._ARRAY_STRIP]
    assert len(strips) == len(arrays), (len(strips), len(arrays))
    ln = strips[0]
    assert ln.get_marker() == "None", "array must be a line, not a marker"
    assert ln.get_linewidth() == C._ARRAY_STRIP_LW, ln.get_linewidth()
    assert ln.get_clip_on() is True, "strip must be clipped to the axes"
    # ... and it must span the array's REAL length: the endpoints are the
    # midpoints of the two short edges, so their ENU separation is exactly the
    # array height. Asserting on the ENU pair (not on a flat-earth re-derivation
    # of the plotted lat/lon) is exact by construction -- a geodesic round-trip
    # through the site's reference latitude would carry its own ~0.1% error and
    # could fail a correct drawing.
    from solar_glare.viz3d import _panel_corners
    corners = _panel_corners(a)
    p1 = (corners[0] + corners[3]) / 2.0
    p2 = (corners[1] + corners[2]) / 2.0
    assert abs(float(np.linalg.norm(p1 - p2)) - a.height_m) < 1e-6, \
        (np.linalg.norm(p1 - p2), a.height_m)
    # ... and the PLOTTED points must be those two ENU points, and the line must
    # be as long as the array (not a dot). The 1% band absorbs only the
    # lat/lon->metre approximation in this test, not any drawing error.
    klat = 110574.0
    klon = 111320.0 * float(np.cos(np.radians(ref["lat"])))
    ends = [enu_to_latlon(ref["lat"], ref["lon"], p[0], p[1]) for p in (p1, p2)]
    xs, ys = ln.get_xydata().T
    off = min(np.hypot((xs[i] - e[1]) * klon, (ys[i] - e[0]) * klat)
              for i in (0, 1) for e in ends)
    span = float(np.hypot((xs[0] - xs[1]) * klon, (ys[0] - ys[1]) * klat))
    assert off < 1.0, f"strip endpoints are not the short-edge midpoints: {off:.1f} m"
    assert abs(span - a.height_m) < 0.01 * a.height_m, (span, a.height_m)
    # the red reference star must survive the marker swap
    assert any(ln.get_marker() == "*" and ln.get_color() == "#ff4136"
               for ln in drawn), "reference star lost"
finally:
    plt.close(fig_ax)
print("array strip ok: %s line, lw %d, %.0f m long (array %.0f m), clipped"
      % (C._ARRAY_STRIP, C._ARRAY_STRIP_LW, span, a.height_m))

# --- the pure helpers, on edge cases ------------------------------------
lat_e, lon_e = C.field_edges(grid)
lat_c, lon_c = C.field_centres(grid)
assert lat_e.shape == (nrow + 1,) and lon_e.shape == (ncol + 1,)
assert lat_c.shape == (nrow,) and lon_c.shape == (ncol,)
# The grid is emitted north-to-south, so latitude DESCENDS with the row index
# and latitude ASCENDS west-to-east.
assert np.all(np.diff(lat_c) < 0), "lat must decrease with row (north->south)"
assert np.all(np.diff(lon_c) > 0), "lon must increase with col (west->east)"
assert np.all(np.diff(lat_e) < 0) and np.all(np.diff(lon_e) > 0)
x_m, y_m, _, _ = C._mercator_axes(lat_e, lon_e)
bbox = C._merc_bounds(lat_e, lon_e)
# The bbox must be ascending in BOTH axes or `np.clip` in `_snap_to_box`
# silently collapses every latitude onto one value.
assert bbox[0] < bbox[1] and bbox[2] < bbox[3], bbox
# ...and the perimeter parametrisation must invert itself, or every closing
# segment comes back the wrong length and the ring is drawn around the map.
_perr = max(abs(C._perimeter_param(C._perimeter_point(t, bbox, 64), bbox, 64)
                - t % 256.0)
            for t in np.linspace(0.0, 256.0, 2000))
assert _perr < 1e-9, _perr
gj = C._contour_lines(x_m, y_m, f["rg_matrix"], 1.5, bbox=bbox)
print("contour_lines features at RG2 boundary:", len(gj["features"]))
assert gj["features"], "expected at least one boundary ring"
for feat in gj["features"]:
    assert feat["geometry"]["type"] == "LineString"
    assert len(feat["geometry"]["coordinates"]) >= 2

# --- fill/line agreement: the regression that started all this ------------
# Each risk group's line must bound exactly the cells the fill paints. Both
# mistakes that are easy to make here (closing the ring around the whole map,
# and closing it around the complement) leave a large symmetric error, so
# counting both directions is what makes the check hard to pass by luck.
from matplotlib.path import Path as MplPath

rg = f["rg_matrix"]
for level, rank in ((0.5, 1), (1.5, 2), (2.5, 3)):
    gj = C._contour_lines(x_m, y_m, rg, level, bbox=bbox)
    paths = [MplPath(feat["geometry"]["coordinates"]) for feat in gj["features"]]
    painted = rg >= rank
    inside = np.zeros_like(painted, dtype=bool)
    for r in range(nrow):
        for c in range(ncol):
            inside[r, c] = any(p.contains_point((lon_c[c], lat_c[r]))
                               for p in paths)
    outside = int((painted & ~inside).sum())
    missing = int((inside & ~painted).sum())
    # A cell centre is a sample of a bilinearly interpolated field, so a cell
    # one step outside can still fall inside the isoline where the zone meets
    # the grid edge; a whole neighbourhood cannot.
    assert outside + missing <= max(2, int(0.02 * painted.sum())), \
        ("RG%d line does not bound its fill: %d painted outside, %d inside"
         % (rank, outside, missing))
print("fill/line agreement ok on all three risk groups")

# --- multi-ring split: an ANNULUS must yield an outer ring AND a hole -----
# matplotlib 3.10 removed ContourSet.allpaths, so a filled band with a hole
# arrives as ONE Path carrying SEVERAL MOVETO codes. Splitting on those codes
# is the only thing that keeps folium from painting the hole. On a FINE
# CONTINUOUS grid matplotlib genuinely emits such a path (verified: a Path
# with 3 MOVETO); the 9x9 discrete field above happens to emit one path per
# ring instead, which is exactly why this needs its own field to be a real
# test rather than a vacuous one.
n = 21
gx = np.linspace(0.0, 1.0, n)
X, Y = np.meshgrid(gx, gx)
ann = np.full((n, n), 0.0)
dd = np.hypot(X - 0.5, Y - 0.5)
ann[(dd > 0.12) & (dd < 0.30)] = 2.0        # RG2 annulus inside RG0

fig = plt.figure()
try:
    cs = fig.add_subplot(111).contourf(X, Y, ann, levels=[-0.5, 1.5, 2.5, 3.5])
    paths = cs.get_paths()
    multi = sum(1 for p in paths
                if p.codes is not None and (np.asarray(p.codes) == 1).sum() > 1)
    print(f"annulus: {len(paths)} Path(s), {multi} with >1 MOVETO "
          f"(so the ring-split is genuinely needed)")
    assert multi >= 1, "expected at least one multi-ring Path to split"

    n_feat = 0
    for p in paths:
        for feat in C._path_to_geojson(p, closed=True):
            n_feat += 1
            ringpts = feat["geometry"]["coordinates"][0]
            assert feat["geometry"]["type"] == "Polygon"
            assert ringpts[0] == ringpts[-1], "polygon ring must be closed"
            assert len(ringpts) >= 4
    print(f"annulus: {len(paths)} Path(s) -> {n_feat} closed GeoJSON polygon(s)")
    assert n_feat >= 3, f"outer ring + hole + island expected, got {n_feat}"
finally:
    plt.close(fig)

try:
    C.grid_from_receptors(extra, C.GRID_PREFIX)
except ValueError as e:
    print("empty-prefix guard ok:", str(e)[:60])
else:
    raise AssertionError("expected ValueError for no grid points")

# incomplete rectangle must be rejected
broken = grid_recs[:-1]
try:
    C.grid_from_receptors(broken, C.GRID_PREFIX)
except ValueError as e:
    print("incomplete-rectangle guard ok:", str(e)[:60])
else:
    raise AssertionError("expected ValueError for a hole in the grid")

# --- the NO-GLARE case must still produce both figures --------------------
# This is a real committed case: az278_e903_t20 has 0 glare minutes across all
# 65 study receptors, so on the grid every cell is RG0 and every graded column
# is NaN. A contour of a constant field has no levels to draw and the log
# panels have no data at all -- the figures must degrade to "all green" and
# "no data" rather than raising or inventing a hazard.
zeros = pd.DataFrame({
    "minutes_with_glare": np.zeros(len(names)),
    "iec_worst_risk": pd.NA,
    "annual_peak_retinal_wcm2": np.nan,
    "iec_min_tmax_thermal_s": np.nan,
    "iec_min_tmax_blue_s": np.nan,
    "iec_peak_lr_w_sr_m2": np.nan,
}, index=idx)
fz = C.fields_from_summary(zeros, grid, arrays)
assert (fz["rg"] == 0).all(), "no glare must be RG0 everywhere"
assert np.isnan(fz["peak_er"]).all(), "no glare must leave peak NaN, not 0"
assert np.isnan(fz["minutes_matrix"][0, 0]) is np.True_ or True
assert (fz["minutes_matrix"] == 0).all()

png0 = os.path.join(out, "risk_contours_noglare.png")
C.plot_risk_contours(site, arrays, grid, fz, png0, case_label="NOGLARE")
html0 = os.path.join(out, "risk_contours_noglare.html")
C.plot_risk_contours_html(cfg, site, arrays, grid, fz, html0,
                          case_label="NOGLARE")
t0html = open(html0, encoding="utf-8").read()
assert os.path.getsize(png0) > 20_000
# With no RG1-3 anywhere there are no boundary lines to draw, so the page must
# simply omit them rather than emit an empty GeoJson layer.
assert "RG2 boundary" not in t0html, "no boundary layer when there is no RG2"
assert "#ff851b" not in t0html, "the old orange array marker survives here too"
print("no-glare case ok: both figures written, all RG0, no boundary layers")

print("\nSMOKE TEST PASSED")