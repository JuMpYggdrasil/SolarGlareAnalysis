"""IEC 62471 isorisk-style contour maps for the Map Ta Phut glare study.

The committed analysis (`config/site.yaml`) evaluates 65 discrete observation
points at 40 m AGL -- the right input for a receptor study, but far too sparse
to draw a contour map: 63 of those 65 points are glare-free in every case, so
interpolating between them would invent the shape of the hazard zones rather
than measure it. `config/site_contour.yaml` therefore adds a ~1 km ENU-aligned
receptor grid at `height_agl_m: 0` (ground level) -- 703 points covering BOTH
glare footprints (east of the array for the az098 cases, west of it for the
az278 ones) -- and runs the UNCHANGED pipeline over it.

Everything in this module is presentation only. A grid point is an ordinary
receptor as far as `analysis.run_analysis` is concerned, so the physics, the
polygon hit detector, the SGHAT hazard classification and the IEC 62471 risk
grouping are exactly the ones the committed runs use. This module only
reshapes the resulting `summary` table into a 2-D field and draws it.

Figures per case, written by `scripts/run_contour.py`:

* ``risk_contours_{a..e}.png`` and ``risk_contours_{a..e}_semi50.png`` -- the
  five static panels, each drawn twice: solid fills, and the same fills at 50%
  alpha. Panel (a) is the isorisk map proper (annual worst IEC risk group,
  filled, with the RG boundaries drawn as contour lines).
* ``risk_contours.html`` -- the same field as an interactive folium map over a
  street basemap, with a query marker on every grid point.
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import (BoundaryNorm, ListedColormap,  # noqa: E402
                                LogNorm, to_rgba)
from matplotlib.patches import Patch  # noqa: E402
import numpy as np
import pandas as pd

from .geometry import enu_to_latlon  # noqa: E402
from .viz import (  # noqa: E402
    _ARRAY_STRIP_COLOR as _ARRAY_STRIP,
    _ARRAY_STRIP_LW,
    _IEC_COLORS,
    _MAP_TILES,
    _MAP_TILES_DEFAULT,
    _RG_LABELS,
    _RG_ORDER,
    _array_footprint_latlon,
    _draw_array_strip,
)

# Receptor-name prefixes written by scripts/_gen_contour_config.py.
GRID_PREFIX = "g_"
VAL_PREFIX = "v_"

# Contour levels for the risk-group field. Grid values are integers 0-3
# (RG0..RG3), so a boundary drawn at 0.5 / 1.5 / 2.5 is exactly the locus where
# the IEC risk group changes -- i.e. the isorisk contour lines.
_RG_LEVELS = [-0.5, 0.5, 1.5, 2.5, 3.5]
_RG_LINES = [0.5, 1.5, 2.5]

# Reuse viz's palette and labels so the contour figures and the committed
# figures 05-11 cannot drift apart in colour or naming. Exempt is RG0.
_RG_RANK = {name: i for i, name in enumerate(_RG_ORDER)}

# RG0 is the SAFE area, so the isorisk panels leave it unpainted instead of
# green: "empty background" is the only thing that reads correctly whether the
# panel sits on white paper or is laid over a street basemap. RG1-3 keep the
# shared viz palette verbatim.
_RG_CLEAR = (0.0, 0.0, 0.0, 0.0)


def _rg_cmap(alpha=1.0):
    """RG0-3 palette for the isorisk panels, with RG0 left as empty background.

    The opacity is BAKED into each RGBA entry rather than handed to
    `contourf(alpha=...)`. That kwarg overrides the alpha channel of *every*
    entry, which turns the transparent RG0 into 50% BLACK in the
    semi-transparent variant -- i.e. it would wash the entire safe area grey,
    exactly where the map is supposed to be blank. Measured, not assumed.
    """
    return ListedColormap(
        [_RG_CLEAR] + [to_rgba(_IEC_COLORS[c], alpha) for c in _RG_ORDER[1:]])


_M_PER_DEG_LAT = 110574.0


# --------------------------------------------------------------------------
# grid extraction and reshaping
# --------------------------------------------------------------------------
def grid_from_receptors(receptors, prefix=GRID_PREFIX):
    """Split a receptor list into the `prefix`-named grid and everything else.

    Returns ``(grid, others)``. ``_gen_contour_config.py`` emits grid points
    north-to-south then west-to-east with the (row, col) index encoded in the
    name, so the summary table can be reshaped back into a 2-D field without
    guessing at a sort order.

    `grid` holds parallel arrays in grid order: ``names``, ``lat``, ``lon``,
    ``row``, ``col``, ``nrow``, ``ncol``, plus ``idx[name] -> position``.
    """
    named = [r for r in receptors if r.name.startswith(prefix)]
    others = [r for r in receptors if not r.name.startswith(prefix)]
    if not named:
        raise ValueError(f"no receptor name starts with {prefix!r}")

    rows, cols = [], []
    for r in named:
        a, _, b = r.name[len(prefix):].partition("_")
        rows.append(int(a))
        cols.append(int(b))
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    nrow, ncol = int(rows.max()) + 1, int(cols.max()) + 1

    if len(named) != nrow * ncol:
        raise ValueError(f"grid is not a complete rectangle: {len(named)} "
                         f"points but {nrow} x {ncol} = {nrow * ncol}")

    order = np.lexsort((cols, rows))            # north-to-south, west-to-east
    named = [named[i] for i in order]
    rows, cols = rows[order], cols[order]

    enu = []
    for r in named:
        e = getattr(r, "enu", None)
        if e is None:
            raise ValueError(f"receptor {r.name!r} has no ENU coordinates; "
                             "build the site first (analysis.build_site)")
        enu.append(e)

    grid = {
        "names": [r.name for r in named],
        "lat": np.array([r.lat for r in named], dtype=float),
        "lon": np.array([r.lon for r in named], dtype=float),
        # Receptor.enu is filled by Receptor.setup() during build_site, so this
        # is the project's own ENU frame (site reference point) rather than a
        # re-projection that could differ in the last digits.
        "enu": np.asarray(enu, dtype=float),
        "row": rows,
        "col": cols,
        "nrow": nrow,
        "ncol": ncol,
        "idx": {r.name: i for i, r in enumerate(named)},
        "height_agl_m": named[0].height_agl_m,
    }
    return grid, others


def rg_rank(values):
    """Map IEC risk-group labels to integers 0-3, for contouring.

    A missing label (receptor with no glare minute at all) becomes RG0, which
    is what `analysis` itself records: `iec_worst_risk` is NA exactly when
    there is nothing to grade.
    """
    out = np.zeros(len(values), dtype=int)
    for i, v in enumerate(values):
        if isinstance(v, str):
            r = _RG_RANK.get(v.strip())
            if r is not None:
                out[i] = r
    return out


def to_matrix(grid, values):
    """Reshape a per-grid-point array into an (nrow, ncol) field.

    Cells with no value stay NaN so `contourf` leaves them unpainted instead of
    extrapolating a level across them.
    """
    flat = np.full(grid["nrow"] * grid["ncol"], np.nan, dtype=float)
    flat[grid["row"] * grid["ncol"] + grid["col"]] = np.asarray(values, dtype=float)
    return flat.reshape(grid["nrow"], grid["ncol"])


def field_centres(grid):
    """1-D lat/lon coordinates of the grid cell CENTRES: (lat[nrow], lon[ncol]).

    This is what matplotlib's `contour`/`contourf` wants -- x must match the
    number of COLUMNS of z and y the number of ROWS, so a (nrow, ncol) field
    takes ncol x-values and nrow y-values, not the ncol+1 / nrow+1 edges that
    `field_edges` returns for the folium raster bounds.
    """
    nrow, ncol = grid["nrow"], grid["ncol"]
    return (grid["lat"].reshape(nrow, ncol)[:, 0].copy(),
            grid["lon"].reshape(nrow, ncol)[0, :].copy())


def field_edges(grid):
    """lat/lon cell EDGES (nrow+1, ncol+1), for the folium raster bounds.

    Using edges rather than centres keeps the drawn map covering the whole
    sampled area instead of shrinking to the inside of the outermost row.

    The step is SIGNED, and the grid runs north-to-south, so dlat is negative.
    That sign matters: with an unsigned step the final edge, lat[-1] + dlat/2,
    would land on the wrong side of the last sample and break monotonicity --
    which silently produced a self-overlapping ImageOverlay bounds box.
    """
    dlat, dlon = grid_spacing_degrees(grid)
    lat0 = grid["lat"].reshape(grid["nrow"], grid["ncol"])[:, 0]
    lon0 = grid["lon"].reshape(grid["nrow"], grid["ncol"])[0, :]
    return (np.concatenate([lat0 - dlat / 2, [lat0[-1] + dlat / 2]]),
            np.concatenate([lon0 - dlon / 2, [lon0[-1] + dlon / 2]]))


def grid_spacing_degrees(grid):
    """Signed median lat/lon spacing of the grid, in degrees.

    Negative dlat because `_gen_contour_config.py` emits rows north-to-south.
    """
    nrow, ncol = grid["nrow"], grid["ncol"]
    lat = grid["lat"].reshape(nrow, ncol)[:, 0]
    lon = grid["lon"].reshape(nrow, ncol)[0, :]
    dlat = float(np.median(np.diff(lat))) if nrow > 1 else 1e-4
    dlon = float(np.median(np.diff(lon))) if ncol > 1 else 1e-4
    return dlat, dlon


def _dx(grid):
    """Half a cell of longitude padding, in degrees."""
    return abs(grid_spacing_degrees(grid)[1]) / 2.0


def _dy(grid):
    """Half a cell of latitude padding, in degrees."""
    return abs(grid_spacing_degrees(grid)[0]) / 2.0


def _aspect(ref):
    """Metric-equal x/y, the convention figures 01/02 already use."""
    return 1.0 / np.cos(np.radians(ref["lat"]))


def _draw_array(ax, arrays, site):
    """Draw each array as one grey strip along its long axis, plus the red star.

    The endpoints of the strip are the MIDPOINTS of the two short edges, i.e.
    the true long-axis ends at the array's own width, so the drawn length is the
    real array length. Clipped to the axes: a 4,890 m strip on a 35 km map runs
    off the panel and would otherwise spill over the figure margin. The strip
    itself is viz._draw_array_strip -- one implementation shared with figures
    01/02 -- only the zorder differs (8 here, to sit over the filled RG bands).
    """
    ref = site["reference"]
    for a in arrays:
        _draw_array_strip(ax, a, ref, zorder=8)
    ax.plot(ref["lon"], ref["lat"], marker="*", ms=13, color="#ff4136",
            mec="k", zorder=8, clip_on=False)


def _axis_style(ax, title, ref, xlabel=True, ylabel=True):
    ax.set_title(title, fontsize=9)
    if xlabel:
        ax.set_xlabel("Longitude (deg E)", fontsize=8)
    if ylabel:
        ax.set_ylabel("Latitude (deg N)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_aspect(_aspect(ref))


# --------------------------------------------------------------------------
# static PNG
# --------------------------------------------------------------------------
def plot_risk_contours(site, arrays, grid, fields, path, case_label="",
                       zoom_km=2.0, alpha=1.0, panel="a", transparent=False,
                       transparent_path=None):
    """One static panel for one case, drawn to `path`.

    `panel` selects which of the five:

    * ``a`` -- the isorisk map proper: annual worst IEC 62471 eye risk group,
      filled with the project's own RG palette and outlined with the RG
      boundary contours.
    * ``b`` -- the same field zoomed onto the RG2-or-worse zone, because at
      ~1 km grid spacing the RG3 core is only one or two cells wide and is
      otherwise lost on a map spanning the whole site.
    * ``c`` / ``d`` / ``e`` -- the continuous fields the RG zones derive from:
      annual glare minutes, thermal ``t_max``, annual peak retinal irradiance.

    `alpha` is the fill opacity, applied to the RISK bands only: RG0 (Exempt)
    is left as empty background in both variants, so the safe area shows the
    panel's own background rather than a wash of colour. Boundary lines, the
    array marker, the reference star, the colorbar and all text stay fully
    opaque, so the two variants differ only in how much of the map shows
    through the RG1-3 fills -- which is what makes them useful laid over each
    other or over a basemap.

    If `transparent_path` is provided, a transparent version is also saved to
    `transparent_path` in a single pass.
    """
    ref = site["reference"]
    lat_c, lon_c = field_centres(grid)
    panel = panel.lower()

    size, dpi = _PANEL_SIZE.get(panel, (8.0, 6.0)), 170
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    suffix = "" if alpha >= 0.999 else "  (fills 50% alpha)"

    if panel == "a":
        cmap = _rg_cmap(alpha)
        norm = BoundaryNorm(_RG_LEVELS, cmap.N)
        filled = ax.contourf(lon_c, lat_c, fields["rg_matrix"],
                             levels=_RG_LEVELS, cmap=cmap, norm=norm,
                             extend="neither")
        lines = ax.contour(lon_c, lat_c, fields["rg_matrix"], levels=_RG_LINES,
                           colors="k", linewidths=1.0)
        ax.clabel(lines, fmt=lambda v: f"RG{int(round(v + 0.5))}",
                  fontsize=8, inline=True)
        ax.set_xlim(lon_c.min() - _dx(grid), lon_c.max() + _dx(grid))
        ax.set_ylim(lat_c.min() - _dy(grid), lat_c.max() + _dy(grid))
        # The RG0 swatch is drawn blank too, so the key still NAMES the empty
        # background rather than leaving the reader to guess what it means.
        ax.legend(handles=[Patch(facecolor=cmap(i), label=_RG_LABELS[c])
                           for i, c in enumerate(_RG_ORDER)],
                  loc="upper right", fontsize=8, title="Risk group",
                  title_fontsize=8, framealpha=0.9)
        _axis_style(ax, f"(a) Isorisk map - annual worst IEC 62471 eye risk "
                        f"group{suffix}", ref)
        cb = fig.colorbar(filled, ax=ax, ticks=[0, 1, 2, 3],
                          fraction=0.022, pad=0.012)
        cb.ax.set_yticklabels([_RG_LABELS[c] for c in _RG_ORDER], fontsize=7)
        cb.set_label("worst risk group over the year", fontsize=8)

    elif panel == "b":
        cmap = _rg_cmap(alpha)
        norm = BoundaryNorm(_RG_LEVELS, cmap.N)
        ax.contourf(lon_c, lat_c, fields["rg_matrix"], levels=_RG_LEVELS,
                    cmap=cmap, norm=norm, extend="neither")
        ax.contour(lon_c, lat_c, fields["rg_matrix"], levels=_RG_LINES,
                   colors="k", linewidths=1.0)
        hot = np.nan_to_num(fields["rg_matrix"], nan=0.0) >= 2
        if hot.any():
            la, lo = fields["lat_matrix"][hot], fields["lon_matrix"][hot]
            pad_lat = zoom_km / _M_PER_DEG_LAT
            pad_lon = zoom_km / (111.320 * np.cos(np.radians(ref["lat"])))
            ax.set_xlim(lo.min() - pad_lon, lo.max() + pad_lon)
            ax.set_ylim(la.min() - pad_lat, la.max() + pad_lat)
        _axis_style(ax, f"(b) Zoom: RG2 and above (ground level){suffix}", ref)

    elif panel == "c":
        _log_contour_panel(fig, ax, lon_c, lat_c, fields["minutes_matrix"],
                           "magma", "glare minutes per year",
                           f"(c) Iso-duration: glare minutes per year{suffix}",
                           ref, floor=1.0, alpha=alpha)

    elif panel == "d":
        _log_contour_panel(fig, ax, lon_c, lat_c, fields["tmax_thermal_matrix"],
                           "cool", "allowed stare time (s)",
                           f"(d) Iso-t_max thermal: allowed stare time (s)"
                           f"{suffix}", ref, floor=1e-3, alpha=alpha)

    elif panel == "e":
        _log_contour_panel(fig, ax, lon_c, lat_c, fields["peak_er_matrix"],
                           "viridis", "annual peak E_r (W/cm$^2$)",
                           f"(e) Iso-irradiance: annual peak retinal "
                           f"(W/cm$^2$){suffix}", ref, floor=1e-3, alpha=alpha)

    else:
        raise ValueError(f"unknown contour panel {panel!r}; expected a-e")

    _draw_array(ax, arrays, site)
    fig.suptitle(f"IEC 62471 isorisk contours - {case_label} - "
                 f"ground level (0 m AGL), ~1 km grid "
                 f"({grid['nrow']}x{grid['ncol']} points)", fontsize=11)
    fig.savefig(path, dpi=dpi, transparent=False,
                facecolor="white",
                bbox_inches="tight")
    if transparent_path:
        fig.savefig(transparent_path, dpi=dpi, transparent=True,
                    facecolor="none",
                    bbox_inches="tight")
    plt.close(fig)


# Canvas per panel. (a) is the full-extent map and wants the room; (b) is the
# zoom, which crops to the RG2+ core; (c)-(e) share one compact square so the
# three continuous fields are directly comparable.
_PANEL_SIZE = {"a": (11.0, 7.0), "b": (7.0, 6.0), "c": (6.5, 6.0),
               "d": (6.5, 6.0), "e": (6.5, 6.0)}

# (suffix, alpha) for each of the two variants written per panel.
_PANEL_VARIANTS = (("", 1.0), ("_semi50", 0.5))


def plot_risk_contour_set(site, arrays, grid, fields, out_dir, case_label="",
                           panels="abcde", zoom_km=2.0, transparent=False):
    """Write every panel in both fill variants with _white/_trans suffixes.

    Files: `risk_contours_a_white.png` + `risk_contours_a_trans.png` for full
    fill, and `risk_contours_a_semi50_white.png` + `risk_contours_a_semi50_trans.png`
    for 50% alpha. All written to `out_dir`.
    """
    import os
    written = []
    os.makedirs(out_dir, exist_ok=True)
    for panel in panels:
        for suffix, alpha in _PANEL_VARIANTS:
            base_name_white = f"risk_contours_{panel}{suffix}_white.png"
            base_name_trans = f"risk_contours_{panel}{suffix}_trans.png"
            path_white = os.path.join(out_dir, base_name_white)
            path_trans = os.path.join(out_dir, base_name_trans) if transparent else None
            plot_risk_contours(site, arrays, grid, fields, path_white,
                               case_label=case_label, zoom_km=zoom_km,
                               alpha=alpha, panel=panel, transparent=False,
                               transparent_path=path_trans)
            written.append(path_white)
            if path_trans:
                written.append(path_trans)
    return written


def _log_contour_panel(fig, ax, lon_e, lat_e, mat, cmap, cbar_label, title,
                       ref, floor=1e-3, alpha=1.0):
    """Filled log-spaced contours of a strictly positive field.

    Non-positive and NaN cells are masked out rather than clipped, so a cell
    with no glare minute is left blank instead of acquiring a fictitious
    minimum-level colour.
    """
    v = np.where(np.isfinite(mat) & (mat > 0), mat, np.nan)
    _axis_style(ax, title, ref)
    if not np.isfinite(v).any():
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#666")
        return
    lo = max(float(np.nanmin(v)), floor)
    hi = float(np.nanmax(v))
    if hi <= lo * 1.01:
        hi = lo * 10.0
    levels = np.logspace(np.log10(lo), np.log10(hi), 7)
    cs = ax.contourf(lon_e, lat_e, v, levels=levels, cmap=cmap,
                     norm=LogNorm(levels[0], levels[-1]), extend="both",
                     alpha=alpha)
    fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03).set_label(cbar_label,
                                                                fontsize=8)


# --------------------------------------------------------------------------
# interactive HTML
# --------------------------------------------------------------------------
def _path_to_geojson(path, closed, props=None):
    """One matplotlib Path (possibly multi-ring) -> GeoJSON features.

    matplotlib 3.10 dropped `ContourSet.allpaths` in favour of `get_paths()`,
    whose Path carries one MOVETO code per ring; splitting on that code is what
    separates a filled band's outer ring from the holes matplotlib punches out
    of an annulus. Emitting each ring as its own feature is what lets folium
    paint the band correctly without needing shapely (not installed here).
    """
    verts = np.asarray(path.vertices, dtype=float)
    codes = (np.asarray(path.codes) if path.codes is not None
             else np.full(len(verts), 2, dtype=np.uint8))
    starts = list(np.flatnonzero(codes == 1))
    if not starts:
        starts = [0]
    bounds = starts + [len(verts)]

    feats = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ring = verts[a:b]
        ring = ring[np.isfinite(ring).all(axis=1)]
        if len(ring) < (3 if closed else 2):
            continue
        pts = [[float(lon), float(lat)] for lon, lat in ring]
        if closed and pts[0] != pts[-1]:
            pts.append(pts[0])
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon" if closed else "LineString",
                         "coordinates": [pts] if closed else pts},
            "properties": dict(props or {}),
        })
    return feats


def _contour_lines(lon_e, lat_e, mat, level):
    """GeoJSON features for the single contour level `level` of `mat`.

    A dedicated figure is created per level because `QuadContourSet.get_paths()`
    merges all levels of one call together; splitting them apart afterwards is
    guesswork, drawing one level at a time is not.
    """
    fig = plt.figure()
    try:
        cs = fig.add_subplot(111).contour(lon_e, lat_e, mat, levels=[level])
        feats = []
        for p in cs.get_paths():
            feats.extend(_path_to_geojson(p, closed=False))
        return {"type": "FeatureCollection", "features": feats}
    finally:
        plt.close(fig)


# Web Mercator sphere radius, the same EPSG:3857 definition Leaflet uses.
_MERC_R = 6378137.0


def _merc_x(lon_deg):
    """Longitude (deg E) -> Web Mercator x (m)."""
    return np.radians(np.asarray(lon_deg, dtype=float)) * _MERC_R


def _merc_y(lat_deg):
    """Latitude (deg N) -> Web Mercator y (m)."""
    return _MERC_R * np.log(np.tan(np.pi / 4
                                   + np.radians(np.asarray(lat_deg,
                                                          dtype=float)) / 2))


def _unmerc_x(x_m):
    return np.degrees(np.asarray(x_m, dtype=float) / _MERC_R)


def _unmerc_y(y_m):
    return np.degrees(2.0 * np.arctan(np.exp(np.asarray(y_m, dtype=float)
                                              / _MERC_R)) - np.pi / 2.0)


def _mercator_axes(lat_e, lon_e):
    """Cell CENTRES as Web Mercator x/y, for contouring the field.

    Returns `(x, y, px_per_cell)` with `x` of length `ncol` and `y` of length
    `nrow`, matching a field's shape -- which is what `contourf`/`contour`
    require. `px_per_cell` is how many image pixels one grid cell should occupy,
    used only to size the figure (see `_contour_bands_png`): rendering the SAME
    native-resolution contour on a larger canvas draws the band polygons at that
    many pixels per cell, so the zone edges come out smooth without inventing
    any resolution the ~1 km data does not have.

    Two bugs in the previous raster are fixed by construction here:

    * **The image must be built in Mercator, not in degrees.** folium/Leaflet
      projects a tile overlay with Web Mercator, so a raster painted in a
      lat/lon box gets stretched differently in x and y. This site's degree box
      aspect is 2.0218 while the old pixel box was 1.9821, i.e. Leaflet squeezed
      the overlay ~2% more in longitude than in latitude, so the coloured cells
      drifted off the vector boundary lines by up to ~0.37 km. Sizing the image
      from the Mercator box aspect (1.9727) makes the stretch isotropic, so the
      raster fill and the vector lines land on the same pixels.
    * **`imshow`'s default extent assumes 1 pixel = 1 data unit.** The old
      `np.kron(..., scale=3)` upsample therefore georeferenced the image 3x too
      large and then had it squeezed back into the same `bounds`, and the saved
      PNG came out 111x56 for a 111x57 request, shifting every row by up to
      0.32 km. Drawing on explicit axes sized `W/dpi x H/dpi` removes both.
    """
    xr, yr = _merc_x(lon_e), _merc_y(lat_e)   # north -> south, so yr descends
    # cell CENTRES derived from the EDGE box: ield_edges returns nrow+1 /
    # ncol+1 values whose first and last sit half a cell OUTSIDE the outermost
    # sample, so averaging adjacent pairs is what puts a centre back onto each
    # sample. Treating the endpoints as centres overshoots the data by half a
    # cell in every direction -- the same off-by-one class that put the old
    # raster 0.32 km out.
    x_c = (xr[:-1] + xr[1:]) / 2
    y_c = (yr[:-1] + yr[1:]) / 2
    # Size the image from the MERCATOR box aspect, not from the cell counts: the
    # grid is not square in Mercator (37x19 cells spanning 1.9727:1), so an
    # image of ncol x nrow cells would still be stretched ~1.3% off axis by
    # Leaflet -- small, but it is exactly the class of error removed here.
    span_x, span_y = abs(xr[-1] - xr[0]), abs(yr[-1] - yr[0])
    width_px = max(2, len(x_c) * _MERC_PX_PER_CELL)
    height_px = max(2, int(round(width_px * span_y / span_x)))
    return x_c, y_c, width_px, height_px


# Image pixels per grid cell. A resolution knob only: it renders the ~1 km zone
# edges smoothly without inventing resolution the data does not have.
_MERC_PX_PER_CELL = 6


def _merc_bounds(lat_e, lon_e):
    """The cell-EDGE box as `(x0, x1, y0, y1)` in Mercator metres, both ascending.

    `lat_e` runs north -> south, so its raw Mercator `y` values DESCEND. Sorting
    them matters beyond tidiness: `_close_ring_to_bbox` clamps with
    `np.clip(y, bbox[2], bbox[3])`, and `np.clip` with `low > high` returns `low`
    for every sample -- which silently flattened every latitude in the ring onto
    one constant and emptied the whole polygon.
    """
    xr, yr = _merc_x(lon_e), _merc_y(lat_e)
    return (float(xr.min()), float(xr.max()), float(yr.min()), float(yr.max()))


def _contour_bands_png(x_m, y_m, field, colors, order, width_px, height_px,
                       dpi=100, alpha=1.0):
    """RG zones as a transparent base64 PNG for folium's `ImageOverlay`.

    This is the SAME `contourf` the static panel (a) makes, on the SAME Mercator
    axes `_contour_lines` contours, so the raster fill and the vector boundary
    lines are two halves of one contour operation and cannot disagree.

    Only levels `[0.5, 1.5, 2.5, 3.5]` are drawn, so rank 0 (RG0, Exempt) is
    never painted at all rather than painted-then-cleared: an unpainted band
    shows the transparent figure background, which is what lets the street
    basemap through over the large safe area.

    The axes span the full cell-EDGE box and the figure is exactly
    `width_px x height_px` px at `dpi`, so a pixel maps to a Mercator distance
    and Leaflet lands the overlay where the lines are. `alpha` is passed straight
    through, so the HTML overlay and the semi-transparent static panel are the
    same picture.
    """
    (x0, x1, y0, y1) = _merc_bounds_ascending(y_m, x_m)
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    try:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("none")
        ax.contourf(x_m, y_m, field, levels=[0.5, 1.5, 2.5, 3.5],
                    colors=[colors[c] for c in order[1:]],
                    alpha=alpha, antialiased=False)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=dpi)
        return ("data:image/png;base64,"
                + base64.b64encode(buf.getvalue()).decode())
    finally:
        plt.close(fig)


def _merc_bounds_ascending(y_m, x_m):
    """Half a cell of padding in Mercator metres, so the axes span the EDGES.

    `x_m` / `y_m` are cell CENTRES, so the plotted extent must be pushed out by
    half a cell on each side for the image to cover the whole sampled area rather
    than stopping at the centre of the outermost cell.
    """
    dx = float(np.mean(np.diff(x_m))) if len(x_m) > 1 else 1000.0
    dy = abs(float(np.mean(np.diff(y_m)))) if len(y_m) > 1 else 1000.0
    return (float(x_m[0]) - dx / 2, float(x_m[-1]) + dx / 2,
            float(y_m[-1]) - dy / 2, float(y_m[0]) + dy / 2)


def _contour_lines(x_m, y_m, field, level, bbox=None):
    """GeoJSON features for the single contour level `level` of `field`.

    Drawn on the SAME Mercator axes as `_contour_bands_png` and converted back
    to lon/lat, so the line is literally the edge of the rendered band.

    A dedicated figure per level is required because `QuadContourSet.get_paths()`
    merges all levels of one call together; splitting them apart afterwards is
    guesswork, drawing one level at a time is not.

    `bbox` is the `(x0, x1, y0, y1)` Mercator box of the cell edges, used to
    close a ring that runs off the edge of the sampled grid (see
    `_close_ring_to_bbox`). The probe point handed to that closer is the mean
    position of the cells at or above `level`, i.e. unambiguously on the
    hazardous side of its own isoline.
    """
    # The sample positions at or above this risk group, in Mercator: what the
    # ring MUST enclose, and what `_close_ring_to_bbox` scores candidates on.
    # These are the hot CELLS, not the cross product of their hot columns and
    # rows -- that product is the hot bounding box, which for a zone hugging one
    # corner is mostly safe ground, so scoring on it picked the complement.
    field = np.asarray(field, dtype=float)
    hot_rows, hot_cols = np.nonzero(np.asarray(field) >= level)
    inside_pts = np.column_stack([x_m[hot_cols], y_m[hot_rows]])
    fig = plt.figure()
    try:
        ax = fig.add_subplot(111)
        cs = ax.contour(x_m, y_m, field, levels=[level])
        feats = []
        for p in cs.get_paths():
            verts = np.asarray(p.vertices, dtype=float)
            if bbox is not None and len(verts) > 1:
                verts = _close_ring_to_bbox(verts, bbox,
                                            inside_pts=inside_pts)
            verts[:, 0] = _unmerc_x(verts[:, 0])
            verts[:, 1] = _unmerc_y(verts[:, 1])
            feats.extend(_path_to_geojson(
                _verts_path(verts), closed=False,
                props={"level": float(level),
                       "label": f"RG{int(round(level + 0.5))}"}))
        return {"type": "FeatureCollection", "features": feats}
    finally:
        plt.close(fig)


def _verts_path(verts):
    """A bare Path of LINETO vertices, so `_path_to_geojson` sees one ring."""
    from matplotlib.path import Path as MplPath
    return MplPath(verts, closed=False)


def _box_corner(i, bbox):
    """Box corner `i` (0..3), counter-clockwise from bottom-left."""
    x0, x1, y0, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]][i % 4])


def _snap_to_box(verts, bbox):
    """Clamp off-box coordinates onto the box. matplotlib can put contour
    vertices a hair outside the sampled extent; left alone they draw a line that
    hangs over the edge of its own fill."""
    out = verts.copy()
    out[:, 0] = np.clip(out[:, 0], bbox[0], bbox[1])
    out[:, 1] = np.clip(out[:, 1], bbox[2], bbox[3])
    return out


def _perimeter_param(pt, bbox, n=512):
    """Where `pt` sits on the bbox perimeter, in units of `n` per edge.

    The box is traced counter-clockwise from bottom-left: `n` steps east along
    the south edge, `n` north along the east edge, `n` west along the north edge,
    `n` south along the west edge -- so the whole perimeter is `0 .. 4n` and a
    stretch of boundary is just an interval on that number line. The point is
    assigned to whichever edge it is closest to (fractions of that edge's own
    length, so the four branches are commensurable).

    Keeping both helpers in this one unit is the whole point: an earlier version
    returned a full-perimeter fraction from here and multiplied by `n` at the
    call site while `_perimeter_point` divided by `n`, so the south edge was
    parametrised 4x short and every "closing segment" came back the wrong length.
    """
    x0, x1, y0, y1 = bbox
    w, h = x1 - x0, y1 - y0
    x, y = float(pt[0]), float(pt[1])
    south = (y - y0) / h
    north = (y1 - y) / h
    west = (x - x0) / w
    east = (x1 - x) / w
    which = min((south, 0), (north, 1), (east, 2), (west, 3))[1]
    if which == 0:
        return west * n
    if which == 1:                                   # north edge, traversed west
        return 2.0 * n + east * n
    if which == 2:                                   # east edge, traversed north
        return n + south * n
    return 3.0 * n + north * n                       # west edge, traversed south


def _perimeter_point(t, bbox, n=512):
    """Inverse of `_perimeter_param`: a perimeter position -> coordinate."""
    x0, x1, y0, y1 = bbox
    w, h = x1 - x0, y1 - y0
    t = t % (4.0 * n)
    if t <= n:
        return (x0 + w * t / n, y0)
    t -= n
    if t <= n:
        return (x1, y0 + h * t / n)
    t -= n
    if t <= n:
        return (x1 - w * t / n, y1)
    t -= n
    return (x0, y1 - h * t / n)


def _close_ring_to_bbox(verts, bbox, inside_pts=None, n=512):
    """Close an open contour ring by following the domain boundary.

    matplotlib does not close a contour against the domain boundary, so a zone
    that runs off the edge of the sampled grid leaves its isoline OPEN -- the
    line stops mid-map while the fill carries on to the edge, so the painted area
    visibly overhangs its own boundary. On the az278 cases 27 of the RG1 cells
    sit on the map edge and the level-0.5 isoline runs straight off it.

    The gap follows ONLY the stretch of boundary between the two open ends, and
    that stretch can be taken two ways round the box -- the short way (the hazard
    zone) or the long way (everything except it). There is no way to tell from
    the geometry alone, so the candidate that captures more of `inside_pts`, the
    actual sample positions at or above this risk group, is chosen. That is the
    same quantity the alignment check measures, so the fix cannot disagree with
    its own acceptance test.

    Inserting all four corners unconditionally, which an earlier version did, is
    NOT that: it always walks the whole perimeter, so a zone touching one edge
    came back ringed around the entire map (497 of 703 cells "inside" a
    200-cell zone).

    Already-closed rings are returned untouched: for an interior zone (RG3) there
    is nothing to add and perturbing a correct line is the worse risk.
    """
    if len(verts) < 3 or np.allclose(verts[0], verts[-1]):
        return _snap_to_box(verts, bbox)
    out = _snap_to_box(verts, bbox)
    if inside_pts is None or len(inside_pts) == 0:
        return np.vstack([out, out[:1]])
    import matplotlib.path as mp
    a = _perimeter_param(out[0], bbox, n)
    b = _perimeter_param(out[-1], bbox, n)
    span = 4.0 * n
    best, best_hits = None, -1
    for direction in (1.0, -1.0):
        # How far it is from the end back round to the start, in this direction.
        # Computed as a single subtraction rather than walked with a `while`
        # against `(t - a) % span`: the endpoints are fractional (contour
        # vertices rarely land on a whole step) so that test never goes exactly
        # zero and the walk could only end on its guard.
        length = (a - b) % span if direction > 0 else (b - a) % span
        nsteps = int(round(length))
        if nsteps <= 0:
            continue
        # `_perimeter_point` wraps internally, so `b + direction * k` may run
        # past `4n` and reappear at the start of the trace.
        walk = [_perimeter_point(b + direction * k, bbox, n)
                for k in range(1, nsteps + 1)]
        cand = np.vstack([out, np.asarray(walk, dtype=float), out[:1]])
        hits = int(mp.Path(cand).contains_points(inside_pts).sum())
        if hits > best_hits:
            best, best_hits = cand, hits
    return best if best is not None else np.vstack([out, out[:1]])


def _popup_html(name, row, col, i, fields):
    def fmt(v, spec=".4g"):
        return spec % v if np.isfinite(v) else "-"
    mins = fields["minutes"][i]
    return (
        f"<b>{name}</b> &nbsp;(row {row}, col {col})<br>"
        f"height 0 m AGL, {fields['dist_m'][i] / 1000.0:.2f} km from array<br>"
        f"<b>IEC 62471 worst risk group: {fields['rg_label'][i]}</b><br>"
        f"glare minutes/year: <b>{int(mins) if np.isfinite(mins) else 0}</b><br>"
        f"annual peak E_r: {fmt(fields['peak_er'][i])} W/cm&sup2;<br>"
        f"t_max thermal: {fmt(fields['tmax_thermal'][i])} s<br>"
        f"t_max blue: {fmt(fields['tmax_blue'][i])} s<br>"
        f"peak L_R: {fmt(fields['peak_lr'][i])} W/(m&sup2;&middot;sr)")


def plot_risk_contours_html(cfg, site, arrays, grid, fields, path,
                            case_label=""):
    """Interactive folium map: RG zones + RG boundaries + per-point query."""
    import folium

    ref = site["reference"]
    tile_key = cfg.get("environment", {}).get("map_tiles", _MAP_TILES_DEFAULT)
    tile = _MAP_TILES.get(tile_key, _MAP_TILES[_MAP_TILES_DEFAULT])
    m = folium.Map(location=[ref["lat"], ref["lon"]], zoom_start=12,
                   tiles=tile["url"], attr=tile["attr"])

    lat_e, lon_e = field_edges(grid)
    x_m, y_m, width_px, height_px = _mercator_axes(lat_e, lon_e)

    folium.raster_layers.ImageOverlay(
        image=_contour_bands_png(x_m, y_m, fields["rg_matrix"], _IEC_COLORS,
                                 _RG_ORDER, width_px, height_px),
        bounds=[[float(lat_e[0]), float(lon_e[0])],
                [float(lat_e[-1]), float(lon_e[-1])]],
        opacity=1.0, name="IEC 62471 risk zones (RG1-3)",
        show=True).add_to(m)

    bbox = _merc_bounds(lat_e, lon_e)
    for level in _RG_LINES:
        gj = _contour_lines(x_m, y_m, fields["rg_matrix"], level, bbox=bbox)
        if not gj["features"]:
            continue
        folium.GeoJson(
            gj, name=f"RG{int(level + 0.5)} boundary", show=True,
            # folium validates style_function by CALLING it with the feature,
            # so it must take one positional argument.
            style_function=lambda feature: {"color": "#000", "weight": 2,
                                            "fillOpacity": 0},
            tooltip=f"RG{int(level + 0.5)} boundary").add_to(m)

    for a in arrays:
        # Same grey-strip convention as the PNGs: outline the footprint in grey
        # at the same weight and with NO fill, so the RG raster underneath stays
        # readable. A 2 m wide polygon collapses to one line at map zoom, which
        # is exactly the strip the static panels draw.
        folium.Polygon(
            _array_footprint_latlon(a, ref), color=_ARRAY_STRIP,
            weight=_ARRAY_STRIP_LW, fill=False,
            tooltip=f"{a.name} ({a.width_m:.0f} x {a.height_m:.0f} m)",
            name="PV array").add_to(m)
    folium.Marker([ref["lat"], ref["lon"]],
                  icon=folium.Icon(color="red", icon="star", prefix="fa"),
                  tooltip="Onshore Connection Point").add_to(m)

    grp = folium.FeatureGroup(
        name=f"Grid query points ({len(grid['names'])})", show=False)
    for i, name in enumerate(grid["names"]):
        folium.CircleMarker(
            [float(grid["lat"][i]), float(grid["lon"][i])], radius=3,
            color="#333", weight=1, fill=True, fill_color="#fff",
            fill_opacity=0.6, tooltip=name,
            popup=folium.Popup(_popup_html(name, int(grid["row"][i]),
                                           int(grid["col"][i]), i, fields),
                               max_width=300)).add_to(grp)
    grp.add_to(m)

    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element(_html_legend(case_label)))
    m.save(path)
    return m


def _html_legend(case_label):
    # RG0 is drawn as an empty background in both the raster overlay and the
    # vector map, so its key swatch is blank (white) rather than green -- the
    # legend has to show the same thing the map shows.
    rows = "".join(
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{"#fff" if c == "Exempt" else _IEC_COLORS[c]};'
        f'border:1px solid #333;margin-right:5px;'
        f'vertical-align:-2px"></span>{_RG_LABELS[c]}<br>'
        for c in _RG_ORDER)
    return f"""
    <div style="position:fixed;bottom:26px;left:12px;z-index:9999;background:#fff;
         padding:8px 12px;border:1px solid #999;border-radius:4px;
         font:12px/1.6 sans-serif;box-shadow:1px 1px 3px rgba(0,0,0,.25)">
      <b>IEC 62471 annual worst risk group</b><br>
      {rows}
      <span style="display:inline-block;width:14px;height:14px;
      background:{_ARRAY_STRIP};border:1px solid #333;margin-right:5px;
      vertical-align:-2px"></span>PV array<br>
      <span style="color:#ff4136">&#9733;</span> Onshore Connection Point<br>
      <span style="color:#666;font-size:11px">{case_label} - ground level 0 m AGL,
      ~1 km grid. Enable &quot;Grid query points&quot; and click a point.</span>
    </div>"""


# --------------------------------------------------------------------------
# summary -> fields
# --------------------------------------------------------------------------
def fields_from_summary(summary, grid, arrays):
    """Per-point and matrix fields, read straight off the pipeline `summary`.

    Only rows whose receptor is one of our grid points are used, and they must
    resolve to a single array: the grid describes one reflective sheet, so a
    multi-array site would need an explicit reduction rule that this function
    deliberately refuses to invent.
    """
    sub = summary[summary.index.get_level_values("receptor").isin(grid["idx"])]
    found = sorted(set(sub.index.get_level_values("array")))
    if found != [arrays[0].name]:
        raise ValueError("contour grid must resolve to a single array; got "
                         f"{found} for array {arrays[0].name!r}")
    sub = sub.xs(arrays[0].name, level="array").reindex(grid["names"])
    if sub["minutes_with_glare"].isna().any():
        missing = [n for n, v in zip(grid["names"], sub["minutes_with_glare"])
                   if np.isnan(v)]
        raise ValueError(f"{len(missing)} grid receptors missing from summary, "
                         f"e.g. {missing[:3]}")

    minutes = sub["minutes_with_glare"].to_numpy(dtype=float)
    rank = rg_rank(sub["iec_worst_risk"].tolist())

    def graded(colname):
        """Peak / t_max columns, NaN wherever there was no glare minute.

        The pipeline leaves them NaN for a glare-free receptor. Mapping them to
        0 instead would draw a spurious zero-level contour through the safe
        area, so the NaN is preserved and the cell is simply left blank.
        """
        v = sub[colname].to_numpy(dtype=float)
        return np.where(minutes > 0, v, np.nan)

    a = arrays[0]
    # Straight-line 3-D distance to the array centre in the site's own ENU
    # frame -- the same quantity the hazard's array-limiting angle uses.
    dist = np.linalg.norm(grid["enu"] - a.center_enu[None, :], axis=1)

    f = {
        "minutes": minutes,
        "rg": rank,
        "rg_label": np.array([_RG_ORDER[r] for r in rank], dtype=object),
        "peak_er": graded("annual_peak_retinal_wcm2"),
        "tmax_thermal": graded("iec_min_tmax_thermal_s"),
        "tmax_blue": graded("iec_min_tmax_blue_s"),
        "peak_lr": graded("iec_peak_lr_w_sr_m2"),
        "dist_m": dist,
    }
    f["rg_matrix"] = to_matrix(grid, rank.astype(float))
    f["minutes_matrix"] = to_matrix(grid, minutes)
    f["peak_er_matrix"] = to_matrix(grid, f["peak_er"])
    f["tmax_thermal_matrix"] = to_matrix(grid, f["tmax_thermal"])
    f["tmax_blue_matrix"] = to_matrix(grid, f["tmax_blue"])
    f["lat_matrix"] = to_matrix(grid, grid["lat"])
    f["lon_matrix"] = to_matrix(grid, grid["lon"])
    return f


_PROVENANCE_COLS = ["peak_elevation_deg", "peak_azimuth_deg", "peak_dni_wm2",
                    "peak_aoi_deg", "peak_reflectivity_used",
                    "peak_subtended_rad", "peak_theta_rad",
                    "peak_atmos_transmittance", "peak_corneal_wcm2"]


def peak_provenance(details, arrays, grid):
    """The per-minute state at each grid point's ANNUAL PEAK retinal minute.

    A contour cell is an aggregate of 525,600 minutes, so a number on the map
    is otherwise unauditable: you cannot tell from `rg_matrix` which AOI or
    which transmittance produced it. This pulls the same row `analysis` calls
    the peak (`idxmax` of `retinal_wcm2`, so it lines up exactly with
    `annual_peak_retinal_wcm2`), which makes every cell checkable against the
    published identity:

        E_c = DNI/1e4 * rho * tau      E_r = E_c * k_f / x**2

    without re-running the physics. Glare-free cells are NaN throughout.
    """
    a_name = arrays[0].name
    out = {c: np.full(len(grid["names"]), np.nan) for c in _PROVENANCE_COLS}
    if not details:
        return out
    for i, name in enumerate(grid["names"]):
        d = details.get((a_name, name))
        if d is None or d.empty:
            continue
        row = d.loc[d["retinal_wcm2"].idxmax()]
        out["peak_elevation_deg"][i] = row.get("elevation_deg", np.nan)
        out["peak_azimuth_deg"][i] = row.get("azimuth_deg", np.nan)
        out["peak_dni_wm2"][i] = row.get("dni_wm2", np.nan)
        out["peak_aoi_deg"][i] = row.get("aoi_deg", np.nan)
        out["peak_reflectivity_used"][i] = row.get("reflectivity_used", np.nan)
        out["peak_subtended_rad"][i] = row.get("subtended_rad", np.nan)
        out["peak_theta_rad"][i] = row.get("theta_rad", np.nan)
        out["peak_atmos_transmittance"][i] = row.get("atmos_transmittance", np.nan)
        out["peak_corneal_wcm2"][i] = row.get("corneal_wcm2", np.nan)
    return out


def contour_grid_csv(grid, fields, arrays, path, provenance=None):
    """Flat per-point table: the machine-readable form of the figures."""
    df = pd.DataFrame({
        "name": grid["names"],
        "row": grid["row"],
        "col": grid["col"],
        "lat": grid["lat"],
        "lon": grid["lon"],
        "height_agl_m": grid["height_agl_m"],
        "array": arrays[0].name,
        "dist_to_array_m": fields["dist_m"],
        "minutes_with_glare": fields["minutes"],
        "iec_worst_risk": fields["rg_label"],
        "iec_rg_rank": fields["rg"],
        "annual_peak_retinal_wcm2": fields["peak_er"],
        "iec_min_tmax_thermal_s": fields["tmax_thermal"],
        "iec_min_tmax_blue_s": fields["tmax_blue"],
        "iec_peak_lr_w_sr_m2": fields["peak_lr"],
    })
    if provenance:
        for col, values in provenance.items():
            df[col] = values
    df.to_csv(path, index=False)
    return df