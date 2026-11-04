"""Report figures (matplotlib PNG) + interactive folium site map."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .hazard import COLORS, CODE_ORDER, flash_threshold, burn_threshold
from .geometry import enu_to_latlon

# Interactive map tile providers (folium). Default "esri_street" is free without
# an API key and avoids the OpenStreetMap Tile Usage Policy User-Agent block
# ("403 Access blocked" for some browsers/webview file:// opens) and the CARTO
# free tier that now asks for an API key. All are free with attribution.
_MAP_TILES = {
    "esri_street": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    },
    "esri": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    },
    "carto": {
        "url": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "attr": ('&copy; <a href="https://www.openstreetmap.org/copyright">'
                 'OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'),
    },
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
}
_MAP_TILES_DEFAULT = "esri_street"

_OBS_NAMES = {"obs_asia", "obs_ptt"}

# Route-stem -> triangle-frame color for the receptor site maps / 3D figures.
# Non-route receptors (obs_*, terminal_*, vehicle_*) keep the default black edge.
# West routes = cool family (teal -> blue -> violet -> purple); east1 = olive
# (warm, so it stays distinct from the west palette).
_ROUTE_COLORS = {
    "west1": "#008080",   # teal
    "west2": "#1565C0",   # blue
    "west3": "#8A2BE2",   # violet
    "west4": "#9C27B0",   # purple
    "east1": "#9A7D0A",   # olive
}


def _dedupe_legend(ax, extra_handles=()):
    """Return (handles, labels) for a legend: artists on `ax` de-duplicated by
    label, plus any `extra_handles` appended (e.g. a fixed-style marker)."""
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    out_h, out_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            out_h.append(h)
            out_l.append(l)
    for h in extra_handles:
        if h.get_label() not in seen:
            seen.add(h.get_label())
            out_h.append(h)
            out_l.append(h.get_label())
    return out_h, out_l


def _obs_legend_handle():
    """Legend handle for the circular observation-point symbol."""
    return Line2D([0], [0], marker="o", ls="", color="w",
                  markerfacecolor="#3d9970", markeredgecolor="k", markersize=8,
                  label="Observation point")


def _key_legend_handle():
    """Legend handle for the triangular 'Key Location' receptor symbol."""
    return Line2D([0], [0], marker="^", ls="", color="w",
                  markerfacecolor="#3d9970", markeredgecolor="k", markersize=8,
                  label="Key Location")


def _array_footprint_latlon(array, ref):
    """Ground footprint (horizontal projection) corners of a PV array as a
    (lat, lon) list.

    Uses the panel corners in ENU and drops the vertical component so the
    projected rectangle keeps the true horizontal scale and orientation.
    """
    from .viz3d import _panel_corners
    corners = _panel_corners(array)
    out = []
    for c in corners:
        lat, lon = enu_to_latlon(ref["lat"], ref["lon"], c[0], c[1])
        out.append([lat, lon])
    return out


# A PV array is drawn as ONE thick strip along its long axis wherever the whole
# array is being LOCATED rather than attributed (figures 01/02 here, and the
# contour figures). The 2 m panel width is far below one pixel at map scale, so
# a filled rectangle or a square marker reads as a dot and hides which way the
# 4,890 m row runs; the line width is a drawing weight, NOT a geographic width.
# Figure 03 is the exception: it draws one strip per sub-panel, coloured by the
# glare minutes that sub-panel wins, and falls back to _ARRAY_STRIP_COLOR.
_ARRAY_STRIP_COLOR = "#8a8a8a"
_ARRAY_STRIP_LW = 6
_ARRAY_STRIP_ALPHA = 0.95


def _draw_array_strip(ax, array, ref, color=_ARRAY_STRIP_COLOR,
                      lw=_ARRAY_STRIP_LW, zorder=2, clip_on=True):
    """Draw `array` as a single strip along its long (height) axis.

    The strip endpoints are the MIDPOINTS of the two short edges, so the drawn
    length is the array's true length rather than its diagonal. `clip_on` is
    True by default: a 4,890 m strip on a 35 km map runs off the panel, and
    unclipped it would spill across the figure margin.
    """
    from .viz3d import _panel_corners
    corners = _panel_corners(array)
    p1 = (corners[0] + corners[3]) / 2.0
    p2 = (corners[1] + corners[2]) / 2.0
    lat1, lon1 = enu_to_latlon(ref["lat"], ref["lon"], p1[0], p1[1])
    lat2, lon2 = enu_to_latlon(ref["lat"], ref["lon"], p2[0], p2[1])
    ax.plot([lon1, lon2], [lat1, lat2], color=color, lw=lw,
            solid_capstyle="butt", alpha=_ARRAY_STRIP_ALPHA, zorder=zorder,
            clip_on=clip_on)


def _code_matrix(long_df, n_days=365):
    """Matrix [day_of_year, half_hour_of_day] of hazard codes for one pair.

    30-min resolution (48 cols); each cell = 30 min x 1 day. Only glare rows
    (hazard != "N") are iterated; non-glare cells stay "N", so this stays fast
    for sub-hourly (per-minute) analysis.

    Kept for sanity_checks.py (which asserts its hazard-Code mapping); figure
    colouring now uses the IEC risk-group matrix `_risk_code_matrix`.
    """
    m = np.full((n_days, 24 * 2), "N", dtype="<U1")
    sub = long_df[long_df["hazard"] != "N"]
    for t, code in zip(sub["time"], sub["hazard"]):
        doy = t.dayofyear - 1
        mod = (t.hour * 60 + t.minute) // 30
        m[doy, mod] = code
    return m


_RG_ORDER = ["Exempt", "RG1", "RG2", "RG3"]  # Exempt == IEC/BS EN "RG0"
_RG_RANK = {"Exempt": 0, "RG1": 1, "RG2": 2, "RG3": 3}
_RG_LABELS = {"Exempt": "RG0 (Exempt)", "RG1": "RG1", "RG2": "RG2", "RG3": "RG3"}


def _require_iec_risk(df):
    """Raise a clear error when RG0-3 colouring is requested without IEC data."""
    if "iec_risk_eye" not in df.columns:
        raise ValueError(
            "risk-group (RG0-3) colouring needs IEC 62471 per-minute output "
            "('iec_risk_eye' column); set environment.iec62471_enabled: true "
            "in config/site.yaml")


def _risk_code_matrix(long_df, n_days=365):
    """Matrix [day_of_year, half_hour_of_day] of IEC risk-group codes.

    30-min resolution (48 cols); each cell = 30 min x 1 day. Cells with no
    glare stay "N" (white); glare rows are coloured by their `iec_risk_eye`
    group (Exempt/RG1/RG2/RG3), with the worst group winning when several
    minutes share a cell. A cell is green (RG0/Exempt) only when real glare was
    detected there yet its intensity is genuinely Exempt.
    """
    _require_iec_risk(long_df)
    m = np.full((n_days, 24 * 2), "N", dtype=object)
    sub = long_df[long_df["hazard"] != "N"]
    for t, rg in zip(sub["time"], sub["iec_risk_eye"]):
        doy = t.dayofyear - 1
        mod = (t.hour * 60 + t.minute) // 30
        prev = m[doy, mod]
        if prev == "N" or _RG_RANK[rg] > _RG_RANK[prev]:
            m[doy, mod] = rg
    return m


def _rg_legend_handles(include_rg0=True):
    """Legend handles for the RG0-3 palette (`_RG_LABELS` labels)."""
    ids = _RG_ORDER if include_rg0 else _RG_ORDER[1:]
    return [Patch(facecolor=_IEC_COLORS[c], label=_RG_LABELS[c]) for c in ids]


def plot_glare_heatmap(long_df, title, path, cmap_norm=None):
    """Classic SGHAT glare occurrence plot: day of year (x) vs half-hour of day (y).

    30-min resolution along the vertical axis; y tick labels stay hourly
    (0:00-23:00 local time). Cells are coloured by IEC 62471 risk group
    (RG0 green -> RG3 red); no-glare cells stay white.
    """
    m = _risk_code_matrix(long_df)
    n_days, n_bins = m.shape
    codes_flat = ["N"] + list(_RG_ORDER)
    data = np.array([[codes_flat.index(c) for c in row] for row in m]).T
    if cmap_norm is None:
        cmap = ListedColormap(["white"] + [_IEC_COLORS[c] for c in _RG_ORDER])
        norm = BoundaryNorm(np.arange(-0.5, len(codes_flat)), cmap.N)
    else:
        cmap, norm = cmap_norm
    month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    month_mid = np.cumsum(month_days) - month_days / 2.0
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                   origin="lower", extent=[0, n_days, 0, n_bins])
    ax.set_xticks(month_mid)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_yticks(np.arange(0, 24 * 2 + 1, 3 * 2))
    ax.set_yticklabels([f"{h}:00" for h in range(0, 25, 3)])
    ax.set_ylabel("Hour of day (local time, UTC+7)")
    ax.set_title(title)
    ax.set_facecolor("white")
    ax.grid(axis="y", color="gray", linewidth=0.4, alpha=0.5)
    handles = [Patch(facecolor="white", edgecolor="0.6", label="No glare")]
    handles += _rg_legend_handles()
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_daily_duration(long_df, title, path):
    """Daily glare duration (min/day) as a stacked RG0-3 bar per day of year."""
    df = long_df[long_df["hazard"] != "N"]
    if df.empty:
        return
    _require_iec_risk(df)
    doy = df["time"].dt.dayofyear.to_numpy()
    rgs = df["iec_risk_eye"].to_numpy()
    n_days = 365
    counts = {c: np.zeros(n_days, dtype=int) for c in _RG_ORDER}
    for d, c in zip(doy, rgs):
        if c in counts:
            counts[c][d - 1] += 1
    month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    month_mid = np.cumsum(month_days) - month_days / 2.0
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(1, n_days + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(n_days)
    for c in _RG_ORDER:
        vals = counts[c]
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, width=1.0, color=_IEC_COLORS[c],
               label=_RG_LABELS[c])
        bottom += vals
    total = bottom
    ymax = max(int(np.nanmax(total)), 60)
    ylim = int(np.ceil(ymax / 15.0) * 15)
    ax.set_xlim(0, n_days)
    ax.set_xticks(month_mid)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylim(0, ylim)
    ax.set_yticks(np.arange(0, ylim + 1, 15))
    ax.set_yticklabels([f"{m}" for m in range(0, ylim + 1, 15)])
    ax.set_ylabel("Daily glare duration (minutes)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_sun_path(pos, title, path, lat=None, lon=None, altitude=0.0):
    """Polar sun-path diagram (zenith vs azimuth) for the whole year.

    Uses the apparent zenith as the radial coordinate, matching the pvlib
    analemma convention (center = zenith / overhead, edge = horizon).

    Input positions are aggregated to one sample per local hour before
    plotting, so minute-resolution inputs still draw a light 8,760-point
    scatter. When `lat`/`lon` are given, also draws the pvlib analemma
    details: hour-of-day labels (at the smallest zenith for each local hour)
    and the sun-path curves for the solstices and equinox (5-min resolution).
    """
    pos = pos.resample("h").first()
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, polar=True)
    zenith = pos["apparent_zenith"].to_numpy()
    az = np.radians(pos["azimuth"].to_numpy())
    ok = zenith < 90
    sc = ax.scatter(az[ok], zenith[ok], c=pos.index.dayofyear[ok].to_numpy(),
                    cmap="viridis", s=6, alpha=0.8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["0", "15", "30", "45", "60", "75", "90"])
    ax.set_xlabel("Solar azimuth (deg from North)")
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.1)
    cb.set_label("Day of year")

    if lat is not None and lon is not None:
        # per-hour analemma ("figure-8") curves + hour labels at the loop apex
        for hour in range(24):
            sub = pos[pos.index.hour == hour].sort_index()
            sub = sub[sub["apparent_zenith"] < 90]
            if sub.empty:
                continue
            az = np.radians(sub["azimuth"].to_numpy())
            zen = sub["apparent_zenith"].to_numpy()
            ax.plot(az, zen, color="0.45", lw=0.8, alpha=0.7, zorder=1)
            p = sub.loc[sub["apparent_zenith"].idxmin()]
            ax.text(np.radians(p["azimuth"]), p["apparent_zenith"] - 4.0, str(hour),
                    fontsize=8, ha="center", va="top", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="0.6", lw=0.5, alpha=0.85))

        # sun-path curves for the solstices and equinox (5-min resolution)
        year = pos.index[0].year
        from .solar import build_times, solar_position
        from .geometry import enu_from_latlon
        for date in pd.to_datetime([f"{year}-03-21", f"{year}-06-21", f"{year}-12-21"]).tz_localize(
                pos.index.tz):
            tt = pd.date_range(date, date + pd.Timedelta("24h"), freq="5min",
                               inclusive="left", tz=pos.index.tz)
            sub = solar_position(tt, lat, lon, altitude=altitude)
            sub = sub[sub["apparent_zenith"] < 90]
            label = date.strftime("%Y-%m-%d")
            ax.plot(np.radians(sub["azimuth"]), sub["apparent_zenith"],
                    lw=1.2, label=label)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_ocular_chart(details, path, legend_path=None):
    """Subtended angle vs retinal irradiance with Sandia hazard-zone overlay.

    The main chart is drawn without a legend; if ``legend_path`` is given the
    legend (hazard zones + receptor labels) is saved as a separate figure.
    """
    xmin_all = 0.6
    ymax_all = 1e2
    for (a, r), d in details.items():
        if not d.empty:
            xmin_all = min(xmin_all, float(d["subtended_rad"].min()))
            ymax_all = max(ymax_all, float(d["retinal_wcm2"].max()))
    # Auto-extend the lower x-limit and upper y-limit to the data (far-field
    # array-limiting angles fall below 1e-3 rad and their sharp retinal image
    # can exceed 1e2 W/cm2; both would otherwise be clipped off-plot).
    xlo = 10.0 ** np.floor(np.log10(xmin_all) - 0.1)
    yhi = 10.0 ** np.ceil(np.log10(ymax_all) + 0.1)
    x = np.geomspace(xlo, 0.6, 400)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.fill_between(x, flash_threshold(x), burn_threshold(x), color="#ffdc00", alpha=0.35,
                    label="Potential after-image (Yellow)")
    ax.fill_between(x, burn_threshold(x), 1e4, color="#ff4136", alpha=0.35,
                    label="Potential eye damage (Red)")
    ax.fill_between(x, 0, flash_threshold(x), color="#2ecc40", alpha=0.30,
                    label="Low hazard (Green)")

    for (a, r), d in details.items():
        if not d.empty:
            ax.scatter(d["subtended_rad"], d["retinal_wcm2"], s=6, alpha=0.5,
                       label=f"{r[:12]} <- {a[:18]}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, 0.6); ax.set_ylim(1e-8, yhi)
    ax.set_xlabel("Subtended source angle (rad)")
    ax.set_ylabel("Retinal irradiance (W/cm2)")
    ax.set_title("Ocular impact: retinal irradiance vs subtended angle")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)

    if legend_path is not None:
        n_entries = 3 + sum(1 for (a, r), d in details.items() if not d.empty)
        ncol = 2 if n_entries > 15 else 1
        nrows = (n_entries + ncol - 1) // ncol
        fig, ax = plt.subplots(figsize=(7.5, 0.45 + 0.32 * nrows))
        ax.axis("off")
        _dummy_fig, _dummy_ax = plt.subplots(figsize=(0.1, 0.1))
        for label, color in [("Low hazard (Green)", "#2ecc40"),
                             ("Potential after-image (Yellow)", "#ffdc00"),
                             ("Potential eye damage (Red)", "#ff4136")]:
            _dummy_ax.plot([], [], color=color, alpha=0.5, label=label)
        for (a, r), d in details.items():
            if not d.empty:
                _dummy_ax.scatter([], [], s=6, alpha=0.5,
                                  label=f"{r[:12]} <- {a[:18]}")
        legend = ax.legend(handles=_dummy_ax.get_legend_handles_labels()[0],
                           labels=_dummy_ax.get_legend_handles_labels()[1],
                           loc="upper left", ncol=ncol, fontsize=7, frameon=True)
        plt.close(_dummy_fig)
        plt.tight_layout()
        fig.savefig(legend_path, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def plot_summary(summary, details, path, iec_enabled=None):
    """Annual hours per IEC 62471 risk group (RG0-3) per receptor (per array).

    Only detected-glare minutes are stacked (each minute split by its
    `iec_risk_eye` group); a green RG0 layer appears only when a receptor has
    genuine glare that is weak enough to be IEC Exempt.  The x-axis lists only
    receptors that actually have glare minutes (zero-glare receptors are
    omitted); an array with no glare at all gets no panel. A zero-glare case
    (``iec_enabled=True`` but no glare minutes anywhere) draws an empty
    placeholder instead of raising.
    """
    rows = []
    for (a, r), d in details.items():
        if d.empty or "iec_risk_eye" not in d.columns:
            continue
        for code, n in d["iec_risk_eye"].value_counts().items():
            rows.append({"array": a, "receptor": r, "code": code, "n": int(n)})
    cnt = pd.DataFrame(rows)
    if cnt.empty:
        if not iec_enabled:
            _require_iec_risk(pd.DataFrame())  # IEC off: raise the clear error
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No glare detected (0 minutes per year)",
                ha="center", va="center", fontsize=12)
        fig.suptitle("Annual IEC 62471 risk-group (RG0-3) hours per observation "
                     "point", y=0.98)
        plt.tight_layout()
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return
    g = summary.reset_index()
    panels = []
    for grp_name in g["array"].unique():
        sub = g[g["array"] == grp_name]
        cg = cnt[cnt["array"] == grp_name] if not cnt.empty else pd.DataFrame()
        glared = set(cg["receptor"])
        sub = sub[sub["receptor"].isin(glared)]
        if sub.empty:
            continue  # no glare on this array: skip the panel entirely
        panels.append((grp_name, sub, cg))
    n_groups = len(panels)
    fig, axes = plt.subplots(1, n_groups, figsize=(6.5 * n_groups, 4.5), sharey=True)
    if n_groups == 1:
        axes = [axes]
    for ax, (grp_name, sub, cg) in zip(axes, panels):
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        for code in _RG_ORDER:
            hours = np.zeros(len(sub))
            if not cg.empty:
                for i, rec in enumerate(sub["receptor"]):
                    hit = cg.loc[(cg["receptor"] == rec) & (cg["code"] == code),
                                 "n"].sum()
                    hours[i] = hit / 60.0
            if hours.sum() <= 0:
                continue
            ax.bar(x, hours, bottom=bottom, color=_IEC_COLORS[code],
                   label=_RG_LABELS[code])
            bottom += hours
        ax.set_xticks(x)
        ax.set_xticklabels(sub["receptor"], rotation=25, ha="right", fontsize=8)
        ax.set_xlabel("Receptor / observation point")
        ax.set_title(grp_name)
        ax.set_ylabel("Hours per year")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    totals = cnt.groupby(["array", "receptor"])["n"].sum() / 60.0
    max_hours = float(totals.max()) if len(totals) else 0.0
    ymax = int(max_hours) + 1
    for ax in axes:
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, color="0.75", ls=":", lw=0.6, alpha=0.8)
    fig.suptitle("Annual IEC 62471 risk-group (RG0-3) hours per observation point",
                 y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


_ROUTE_MARK = "_route_"


def _route_stem_index(name):
    """Split a route receptor name into (stem, index); return (None, None) for
    non-route names (obs_*, terminal_*, vehicle_*)."""
    if _ROUTE_MARK not in name:
        return None, None
    stem, rest = name.split(_ROUTE_MARK, 1)
    digits = ""
    for ch in rest:
        if ch.isdigit():
            digits += ch
        else:
            break
    return (stem, int(digits)) if digits else (None, None)


def _route_endpoint_names(receptors):
    """Names of the first and last point of every route, so dense route points
    (e.g. the ~2 km west1-4 sampling) only get endpoint text labels."""
    groups = {}
    for r in receptors:
        stem, idx = _route_stem_index(r.name)
        if stem is not None:
            groups.setdefault(stem, []).append((idx, r.name))
    out = set()
    for items in groups.values():
        items.sort()
        out.add(items[0][1])
        out.add(items[-1][1])
    return out


def _route_color(name):
    """Triangle-frame (edge) color for a route receptor name; 'k' for any
    non-route name (obs_*, terminal_*, vehicle_*)."""
    stem, _idx = _route_stem_index(name)
    return _ROUTE_COLORS.get(stem, "k")


def _route_legend_handles():
    """Route-color triangle handles for the legend (frame = route color,
    fill = default receptor green so the frame color reads clearly)."""
    out = []
    for stem, color in _ROUTE_COLORS.items():
        out.append(Line2D([0], [0], marker="^", ls="", color="w",
                          markerfacecolor="#3d9970", markeredgecolor=color,
                          markersize=8, linewidth=1.3, label=f"Route {stem}"))
    return out


def plot_site_map(cfg, site, arrays, receptors, path, transparent=True):
    """Offline static site map (matplotlib). `transparent=True` saves an RGBA
    PNG with a transparent background; `False` uses an opaque white one."""
    ref = site["reference"]
    fig, ax = plt.subplots(figsize=(8, 7))
    ch = cfg.get("channel", {})
    pts = np.array(ch.get("points", []))
    if pts.size:
        ax.plot(pts[:, 1], pts[:, 0], "--", color="#00509e", lw=2, label="Approach channel (demo)")
    for a in arrays:
        # Strip first (zorder 2) so the orange square marker (zorder 3) sits
        # ON TOP of it: the square keeps the legend, the strip shows the row's
        # true extent and orientation. The legend entry is unchanged.
        _draw_array_strip(ax, a, ref)
        ax.scatter(a.lon, a.lat, marker="s", s=90, c="#ff851b", edgecolor="k", zorder=3)
        ax.annotate(a.name.split("(")[0], (a.lon, a.lat), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color="#7f4f00")
    _hidden = {"terminal_1_west", "terminal_2_east"}
    _route_ep = _route_endpoint_names(receptors)
    for r in receptors:
        marker = "o" if r.name in _OBS_NAMES else "^"
        ax.scatter(r.lon, r.lat, marker=marker, s=70, c="#3d9970",
                   edgecolor=_route_color(r.name), linewidths=1.3, zorder=3)
        if r.name not in _hidden:
            stem, _idx = _route_stem_index(r.name)
            if stem is not None and r.name not in _route_ep:
                continue
            ax.annotate(r.name, (r.lon, r.lat), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
    ax.scatter(ref["lon"], ref["lat"], marker="*", s=160, c="#ff4136", edgecolor="k",
               zorder=4, label="Onshore Connection Point")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    site_label = site["name"].rsplit("(", 1)[0].strip() if site["name"].endswith(")") else site["name"]
    ax.set_title(f"Map Ta Phut area - PV arrays and observation points\n({site_label})")
    extra = ([_obs_legend_handle(), _key_legend_handle()]
             if any(r.name in _OBS_NAMES for r in receptors) else [])
    extra += _route_legend_handles()
    handles, labels = _dedupe_legend(ax, extra)
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    # metric-equal x/y (same as plot_site_map_hazard): 1 deg lon is shorter
    # than 1 deg lat by cos(lat); keeps 01/02 at the same geographic scale.
    ax.set_aspect(1.0 / np.cos(np.radians(ref["lat"])))
    plt.tight_layout()
    fig.savefig(path, dpi=200, transparent=transparent)
    plt.close(fig)


def plot_site_map_folium(cfg, site, arrays, receptors, path):
    """Interactive HTML map (folium). Requires internet for tiles."""
    import folium

    ref = site["reference"]
    tile_key = cfg.get("environment", {}).get("map_tiles", _MAP_TILES_DEFAULT)
    tile = _MAP_TILES.get(tile_key, _MAP_TILES[_MAP_TILES_DEFAULT])
    m = folium.Map(location=[ref["lat"], ref["lon"]], zoom_start=13,
                   tiles=tile["url"], attr=tile["attr"])
    ch = cfg.get("channel", {})
    pts = ch.get("points", [])
    if pts:
        folium.PolyLine(pts, color="blue", weight=3, dash_array="6", tooltip=ch.get("name", "")).add_to(m)
    for a in arrays:
        groups = getattr(a, "group_arrays", None) or []
        panels = groups if len(groups) > 1 else [a]
        for p in panels:
            footprint = _array_footprint_latlon(p, ref)
            folium.Polygon(footprint, color="#ff851b", weight=2, fill=True,
                           fill_color="#ff851b", fill_opacity=0.5,
                           tooltip=f"{p.name} ({p.width_m:.0f}x{p.height_m:.0f} m)").add_to(m)
            if len(groups) > 1:
                glat, glon = enu_to_latlon(ref["lat"], ref["lon"],
                                           p.center_enu[0], p.center_enu[1])
                folium.CircleMarker([glat, glon], radius=8, color="#d9480f",
                                    fill=True, fill_color="#ff851b", fill_opacity=0.9,
                                    tooltip=f"Group center: {p.name}").add_to(m)
        folium.Marker([a.lat, a.lon], icon=folium.Icon(color="orange", icon="solar-panel", prefix="fa"),
                      tooltip=a.name).add_to(m)
    for r in receptors:
        if r.name in _OBS_NAMES:
            folium.CircleMarker([r.lat, r.lon], radius=7, color="#3d9970",
                                fill=True, fill_color="#3d9970", fill_opacity=1.0,
                                tooltip=f"{r.name} (observation point)").add_to(m)
        else:
            folium.Marker([r.lat, r.lon], icon=folium.Icon(color="green", icon="eye", prefix="fa"),
                          tooltip=r.name).add_to(m)
    folium.Marker([ref["lat"], ref["lon"]], icon=folium.Icon(color="red", icon="star", prefix="fa"),
                  tooltip="Onshore Connection Point").add_to(m)
    legend_html = """
    <div style="position:fixed;bottom:40px;left:12px;z-index:9999;background:#fff;
         padding:6px 10px;border:1px solid #999;border-radius:4px;
         font:12px/1.5 sans-serif;box-shadow:1px 1px 3px rgba(0,0,0,.2)">
      <b>Legend</b><br>
      <span style="color:#3d9970">&#9679;</span> Observation point<br>
      <span style="color:#fff;background:#3d9970;border:1px solid #000">&#9650;&nbsp;Key Location</span><br>
      <span style="color:#ff851b">&#9632;</span> PV array &nbsp;
      <span style="color:#ff4136">&#9733;</span> Onshore Connection Point
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(path)
    return m


def _fmt_minutes(mins):
    """Format a minute count as '45 min' or '1h 40m' when >= 60 min."""
    m = int(round(mins))
    if m < 60:
        return f"{m} min"
    return f"{m // 60}h {m % 60:02d}m"


def plot_site_map_hazard(cfg, site, arrays, receptors, summary, path):
    """Site map coloured by receptor glare-minutes (G+Y+R total) with the value
    annotated on each receptor that has glare. Receptors with zero glare minutes
    stay grey."""
    ref = site["reference"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ch = cfg.get("channel", {})
    pts = np.array(ch.get("points", []))
    if pts.size:
        ax.plot(pts[:, 1], pts[:, 0], "--", color="#00509e", lw=2, label="Approach channel (demo)")
    for a in arrays:
        # Same grey strip as figure 01, drawn first so the orange square (zorder
        # 3) stays on top. See plot_site_map.
        _draw_array_strip(ax, a, ref)
        ax.scatter(a.lon, a.lat, marker="s", s=120, c="#ff851b", edgecolor="k", zorder=3)
        ax.annotate(a.name.split("(")[0], (a.lon, a.lat), textcoords="offset points",
                    xytext=(10, 10), fontsize=9, color="#7f4f00")

    per_receptor = summary.reset_index().groupby("receptor")[["G", "Y", "R"]].sum()
    minutes = per_receptor.sum(axis=1)
    vmax = max(float(minutes.max()), 1.0)
    for r in receptors:
        h = float(minutes.get(r.name, 0.0))
        if h > 0:
            frac = h / vmax
            color = plt.cm.cool(0.15 + 0.85 * frac)
        else:
            color = "#cccccc"
        marker = "o" if r.name in _OBS_NAMES else "^"
        ax.scatter(r.lon, r.lat, marker=marker, s=110, c=[color],
                   edgecolor=_route_color(r.name), linewidths=1.3, zorder=3)
        if h > 0:
            ax.annotate(_fmt_minutes(h), (r.lon, r.lat), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, ha="left", zorder=4)
    ax.scatter(ref["lon"], ref["lat"], marker="*", s=180, c="#ff4136", edgecolor="k",
               zorder=4, label="Onshore Connection Point")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("Map Ta Phut area - glare hazard minutes per observation point")
    extra = ([_obs_legend_handle(), _key_legend_handle()]
             if any(r.name in _OBS_NAMES for r in receptors) else [])
    extra += _route_legend_handles()
    handles, labels = _dedupe_legend(ax, extra)
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    # metric-equal x/y: 1 deg lon is shorter than 1 deg lat by cos(lat); without
    # this the 1000-m ring around the array renders as an ellipse.
    ax.set_aspect(1.0 / np.cos(np.radians(ref["lat"])))
    sm = plt.cm.ScalarMappable(cmap="cool",
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
    cb.set_label("Glare minutes per year (G+Y+R)")
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_group_glare_map(cfg, site, arrays, receptors, details, path, receptor=None):
    """Site map with each array's `groups` sub-panel shown as a coloured strip
    along its height axis, coloured by the annual glare minutes attributed to
    that sub-panel (the true 2 m-wide footprints are sub-pixel at map scale,
    so strips are thickened and not to geographic width). Strip colour uses the
    PuRd (purple-pink / lavender) ramp.

    Attribution source: the worst-wins `group` column in the per-pair detail
    tables (``PV-A#<i>``, written by ``analyze_pair``). Groups that never win a
    minute stay grey, so the figure shows which part of the reflective sheet
    actually produces the glare (group 0 = start of the array's height axis,
    the last index = its end, per ``_split_groups``). Receptor markers are drawn
    in the standard style for geographic context only.

    `receptor` (optional): when given, only glare minutes seen by that one
    receptor are counted/coloured (per-receptor colour scale), its marker is
    highlighted with a red ring, and the title names the receptor. When None the
    figure aggregates every receptor pair (per-array scale) as before.
    """
    from .viz3d import _panel_corners
    ref = site["reference"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ch = cfg.get("channel", {})
    pts = np.array(ch.get("points", []))
    if pts.size:
        ax.plot(pts[:, 1], pts[:, 0], "--", color="#00509e", lw=2,
                label="Approach channel (demo)")

    # Per-group total glare minutes across the selected (array, receptor) pairs.
    group_minutes = {}
    for (a, r), d in details.items():
        if receptor is not None and r != receptor:
            continue
        if d.empty or "group" not in d.columns:
            continue
        for grp, cnt in d["group"].value_counts().items():
            group_minutes[grp] = group_minutes.get(grp, 0) + int(cnt)

    # Each array's own scale (avoids one huge colour bar drowning everyone).
    vmaxes = {}
    for a in arrays:
        groups = getattr(a, "group_arrays", None) or [a]
        vmaxes[a.name] = max((group_minutes.get(g.name, 0) for g in groups),
                             default=0.0) or 1.0

    for a in arrays:
        groups = getattr(a, "group_arrays", None) or [a]
        for gi, gsrc in enumerate(groups):
            corners = _panel_corners(gsrc)
            # The 2 m panel width is far below one pixel at map scale, so a
            # filled rectangle renders as an invisible sub-pixel sliver (only
            # the dark edge stroke survives). Draw each sub-panel instead as a
            # thick strip along its height (long) axis, coloured by the glare
            # minutes that group wins (strip width not to geographic scale).
            # Long-axis endpoints = midpoints of the +s / -s edges.
            p1 = (corners[0] + corners[3]) / 2.0
            p2 = (corners[1] + corners[2]) / 2.0
            e1 = enu_to_latlon(ref["lat"], ref["lon"], p1[0], p1[1])
            e2 = enu_to_latlon(ref["lat"], ref["lon"], p2[0], p2[1])
            mn = group_minutes.get(gsrc.name, 0)
            if mn > 0:
                color = plt.cm.PuRd(0.15 + 0.85 * mn / vmaxes[a.name])
            else:
                # NB: a DATA colour, not _ARRAY_STRIP_COLOR. Here grey means
                # "this sub-panel wins zero glare minutes"; over there it means
                # "this is the array". Keeping them separate stops figure 03
                # from darkening its zero-glare groups.
                color = "#cccccc"
            ax.plot([e1[1], e2[1]], [e1[0], e2[0]], color=color, lw=6,
                    solid_capstyle="butt", zorder=3, alpha=0.95)
            if mn > 0:
                # Label sits west of the strip (screen-left) so it does not
                # cover the coloured line.
                cxlat, cxlon = enu_to_latlon(
                    ref["lat"], ref["lon"],
                    gsrc.center_enu[0] - 1400.0, gsrc.center_enu[1])
                ax.annotate(f"#{gi} {_fmt_minutes(mn)}", (cxlon, cxlat),
                            ha="center", va="center", fontsize=7, zorder=5,
                            color="#111111")

    # Receptors (neutral green; glare-minute values live in fig 02).
    for r in receptors:
        marker = "o" if r.name in _OBS_NAMES else "^"
        ax.scatter(r.lon, r.lat, marker=marker, s=70, c="#3d9970",
                   edgecolor=_route_color(r.name), linewidths=1.3, zorder=6)
    ax.scatter(ref["lon"], ref["lat"], marker="*", s=180, c="#ff4136",
               edgecolor="k", zorder=7, label="Onshore Connection Point")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    if receptor is not None:
        # Highlight ring around the receptor of interest.
        tr = next(r for r in receptors if r.name == receptor)
        marker = "o" if tr.name in _OBS_NAMES else "^"
        ax.scatter(tr.lon, tr.lat, marker=marker, s=150, facecolors="none",
                   edgecolors="#ff4136", linewidths=2.0, zorder=8,
                   label=f"Receptor (glare): {receptor}")
        ax.set_title(f"Map Ta Phut - glare-producing sub-panels as seen from "
                     f"{receptor}")
    else:
        ax.set_title("Map Ta Phut - glare-producing sub-panels (groups)")
    extra = ([_obs_legend_handle(), _key_legend_handle()]
             if any(r.name in _OBS_NAMES for r in receptors) else [])
    extra += _route_legend_handles()
    handles, labels = _dedupe_legend(ax, extra)
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect(1.0 / np.cos(np.radians(ref["lat"])))
    sm = plt.cm.ScalarMappable(cmap="PuRd",
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
    cb.set_label(("Glare minutes per sub-panel for "
                  f"{receptor}" if receptor else
                  "Glare minutes per sub-panel (G+Y+R)") + ", per-array scale")
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_sun_chart(details, path, iec_enabled=None):
    """Scatter of glare events by solar azimuth/elevation per IEC risk group."""
    any_iec = any(not d.empty and "iec_risk_eye" in d.columns for d in details.values())
    if not any_iec and not iec_enabled:
        _require_iec_risk(pd.DataFrame())
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 6.5))
    for rg, c in _IEC_COLORS.items():
        xs, ys = [], []
        for d in details.values():
            if d.empty or "iec_risk_eye" not in d.columns:
                continue
            sub = d[d["iec_risk_eye"] == rg]
            if sub.empty:
                continue
            xs.append(np.radians(sub["azimuth_deg"].to_numpy()))
            ys.append(sub["elevation_deg"].to_numpy())
        if not xs:
            continue
        ax.scatter(np.concatenate(xs), np.concatenate(ys), s=8, alpha=0.7,
                   color=c, label=_RG_LABELS[rg])
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_ylim(0, 90); ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(["0", "30", "60", "90"])
    ax.set_title("Glare events by sun position - IEC 62471 risk group (RG0-3)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# IEC 62471 chart helpers (require ``iec62471_enabled: true`` in config)
# ---------------------------------------------------------------------------

_IEC_COLORS = {
    "Exempt": "#2ecc40",  # RG0 (no/negligible hazard) - green
    "RG1":    "#9acd32",  # yellow-green
    "RG2":    "#ffdc00",  # yellow
    "RG3":    "#ff4136",  # red
}


def plot_iec62471_zones(details, path):
    """IEC 62471 thermal hazard zones: log-log scatter of alpha vs L_R.

    Overlayed zone boundaries are from BS EN 62471:2008 Table 6.1 for the
    reference duration of 10 s (Exempt / RG1 = 2.8e4 / alpha) and 0.25 s
    (RG2 = 7.1e4 / alpha).  Points are coloured by formal eye risk group.
    Only pairs whose details carry IEC columns are plotted.
    """
    a_all, lr_all = [], []
    for d in details.values():
        if d.empty or "iec_alpha_rad" not in d.columns:
            continue
        a_all.append(d["iec_alpha_rad"])
        lr_all.append(d["iec_lr_w_sr_m2"])
    if not a_all:
        return  # nothing to draw
    a_all = np.concatenate(a_all)
    lr_all = np.concatenate(lr_all)
    xlo = 10 ** np.floor(np.log10(max(a_all.min(), 1e-4)))
    xhi = 10 ** np.ceil(np.log10(max(a_all.max(), 1e-2)))
    ylo = 10 ** np.floor(np.log10(max(lr_all.min(), 1e2)))
    yhi = 10 ** np.ceil(np.log10(max(lr_all.max(), 1e8)))
    x = np.geomspace(xlo, xhi, 400)
    fig, ax = plt.subplots(figsize=(8, 6))
    # zone shading (bottom-up)
    ax.fill_between(x, yhi, 7.1e4 / x, color=_IEC_COLORS["RG3"], alpha=0.18,
                    label="RG3 (High Risk)")
    ax.fill_between(x, 7.1e4 / x, 2.8e4 / x, color=_IEC_COLORS["RG2"], alpha=0.18,
                    label="RG2")
    ax.fill_between(x, 2.8e4 / x, 10 ** np.floor(np.log10(ylo)),
                    color=_IEC_COLORS["RG1"], alpha=0.18, label="Exempt / RG1")
    ax.plot(x, 2.8e4 / x, color=_IEC_COLORS["RG1"], lw=1.5, ls="--",
            label="Exempt / RG1 boundary (10 s)")
    ax.plot(x, 7.1e4 / x, color=_IEC_COLORS["RG2"], lw=1.5, ls="--",
            label="RG2 boundary (0.25 s)")
    # scatter per-risk-coloured point
    for rname in _IEC_COLORS:
        mask = np.array([False])
        for (a, r), d in details.items():
            if d.empty or "iec_risk_eye" not in d.columns:
                continue
            m = d["iec_risk_eye"] == rname
            if m.any():
                ax.scatter(d.loc[m, "iec_alpha_rad"], d.loc[m, "iec_lr_w_sr_m2"],
                           s=10, alpha=0.7, color=_IEC_COLORS[rname],
                           label=f"Glare minutes: {rname}" if not mask.any() else None)
                mask = np.ones(1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Apparent source angle  alpha (rad)")
    ax.set_ylabel("R-weighted radiance  L_R  (W m$^{-2}$ sr$^{-1}$)")
    ax.set_title("IEC 62471 retinal thermal hazard zones")
    ax.grid(True, which="both", alpha=0.25)
    handles, labels = _dedupe_legend(ax)
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8,
              title="Zone / risk")
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_iec62471_tmax(details, thermal_path, blue_path):
    """Allowed exposure duration t_max: one chart per hazard clause.

    ``thermal_path`` receives the thermal-retinal chart (RG transitions at
    0.25 s / 10 s, circle markers), ``blue_path`` the blue-light chart
    (1 s / 100 s, triangle markers).  Each chart shades the RG zones as
    horizontal bands (alpha 0.18, like fig 10) so the point colours and the
    transition lines agree within a single figure.
    """
    _plot_iec62471_tmax_mode(details, "thermal", thermal_path)
    _plot_iec62471_tmax_mode(details, "blue", blue_path)


def _plot_iec62471_tmax_mode(details, mode, path):
    """Helper: one IEC 62471 t_max log-log chart for ``mode`` ("thermal"/"blue")."""
    t_col = "iec_tmax_thermal_s" if mode == "thermal" else "iec_tmax_blue_s"
    marker = "o" if mode == "thermal" else "^"
    line_lw = 3.0 if mode == "thermal" else 2.0
    line_alpha = 0.7 if mode == "thermal" else 0.5
    line_ls = "--" if mode == "thermal" else ":"
    t_hi, t_lo = (10.0, 0.25) if mode == "thermal" else (100.0, 1.0)
    line_specs = [(t_hi, _IEC_COLORS["RG1"], f"{mode} t = {t_hi:g} s"),
                  (t_lo, _IEC_COLORS["RG2"], f"{mode} t = {t_lo:g} s")]
    a_arr, t_arr = [], []
    for d in details.values():
        if d.empty or "iec_alpha_rad" not in d.columns:
            continue
        mask = d[t_col].notna()
        if mask.any():
            a_arr.append(d.loc[mask, "iec_alpha_rad"].to_numpy())
            t_arr.append(d.loc[mask, t_col].to_numpy())
    if not a_arr:
        return  # nothing to draw
    a_arr = np.concatenate(a_arr)
    t_arr = np.concatenate(t_arr)
    xlo = 10 ** np.floor(np.log10(max(a_arr.min(), 1e-4)))
    xhi = 10 ** np.ceil(np.log10(max(a_arr.max(), 1e-2)))
    ylo = 1e-2  # fixed y-axis floor (0.01 s) for both clauses, never clips data
    yhi = 10 ** np.ceil(np.log10(max(t_arr.max(), 1e3)))
    x = np.geomspace(xlo, xhi, 400)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    # zone shading (horizontal bands RG1 -> RG2 -> RG3, like fig 10)
    zone_specs = [
        (yhi, t_hi, _IEC_COLORS["RG1"], f"RG1 zone (t >= {t_hi:g} s)"),
        (t_hi, t_lo, _IEC_COLORS["RG2"], f"RG2 zone ({t_lo:g} - {t_hi:g} s)"),
        (t_lo, ylo, _IEC_COLORS["RG3"], f"RG3 zone (t < {t_lo:g} s)"),
    ]
    for up, lo_, c, lbl in zone_specs:
        lo = max(lo_, ylo)
        hi = min(up, yhi)
        if hi > lo:
            ax.fill_between(x, hi, lo, color=c, alpha=0.18, label=lbl)
    for tval, col, lbl in line_specs:
        ax.axhline(tval, color=col, lw=line_lw, ls=line_ls, alpha=line_alpha,
                   label=lbl)
    mask_rg3 = t_arr < t_lo
    mask_rg2 = (t_arr >= t_lo) & (t_arr < t_hi)
    mask_rg1 = t_arr >= t_hi
    for m, c, lbl in [
        (mask_rg3, _IEC_COLORS["RG3"], f"RG3 ({mode})"),
        (mask_rg2, _IEC_COLORS["RG2"], f"RG2 ({mode})"),
        (mask_rg1, _IEC_COLORS["RG1"], f"RG1 ({mode})"),
    ]:
        if m.any():
            ax.scatter(a_arr[m], t_arr[m], s=10, alpha=0.65, color=c,
                       marker=marker, label=lbl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Apparent source angle  alpha (rad)")
    ax.set_ylabel("Allowed exposure duration  t$_{max}$ (s)")
    ax.set_title(f"IEC 62471 allowed exposure duration - {mode} ({marker})")
    ax.grid(True, which="both", alpha=0.25)
    handles, labels = _dedupe_legend(ax)
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8,
              title=f"{mode.capitalize()} thresholds")
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_iec62471_weights(path):
    """IEC 62471 Figure 4.2: spectral weighting functions B(lambda) and R(lambda).

    x = 300-1400 nm, log ordinate 1e-4..1e1 exactly like the standard's figure.
    Curves are the same Table 4.2 functions used by the k_X computation, so the
    figure is always consistent with the pipeline's k_R/k_B.
    """
    from solar_glare import iec62471 as _iec

    wl = np.linspace(300.0, 1400.0, 2201)
    b = np.array([_iec._b_weight(x) for x in wl])
    r = np.array([_iec._r_weight(x) for x in wl])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(wl, r, color="#2b6cb0", lw=2.0, ls="-", label="$R(\\lambda)$ burn function")
    ax.semilogy(wl, b, color="#c05621", lw=2.0, ls="--", label="$B(\\lambda)$ blue-light function")
    ax.set_xlim(300.0, 1400.0)
    ax.set_ylim(1e-4, 1e1)
    ax.set_xlabel("Wavelength $\\lambda$ (nm)")
    ax.set_ylabel("Spectral weighting")
    ax.set_title("IEC 62471 Figure 4.2 -- spectral weighting functions for retinal hazards")
    ax.grid(True, which="both", alpha=0.25)
    handles, labels = _dedupe_legend(ax)
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8,
              title="Weighting function")
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_iec62471_spectrum_weighted(path):
    """Reference solar spectrum before and after the IEC 62471 weighting.

    Two stacked panels sharing the wavelength axis:
      (a) the raw IEC 60904-3:2019 Table 1 direct-normal spectrum S(lambda),
          log-x over 280-4000 nm, with the integrated DNI annotated;
      (b) the weighted spectra S(lambda)*B(lambda) and S(lambda)*R(lambda),
          log-y over 300-1400 nm, showing which wavelengths actually drive
          k_B and k_R (the 435-440 nm blue-violet hump dominates R).
    Uses the exact _spectrum()/_b_weight()/_r_weight() functions of the
    pipeline, so the figure is always consistent with the computed k_X.
    """
    from solar_glare import iec62471 as _iec

    wl_s, e_s = _iec._spectrum()
    wl = np.linspace(300.0, 1400.0, 2201)
    b = np.array([_iec._b_weight(x) for x in wl])
    r = np.array([_iec._r_weight(x) for x in wl])

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    axa.semilogx(wl_s, e_s, color="#2c2c2c", lw=1.2)
    axa.set_xlim(280.0, 4000.0)
    axa.set_xlabel("Wavelength $\\lambda$ (nm)")
    axa.set_ylabel("Irradiance (W m$^{-2}$ nm$^{-1}$)")
    axa.set_title("Reference spectrum (before weighting) -- IEC 60904-3:2019 Table 1 Direct")
    axa.grid(True, which="both", alpha=0.25)
    dni = float(np.trapezoid(e_s, wl_s))
    axa.text(0.03, 0.93,
             f"Integrated direct irradiance = {dni:.0f} W m$^{{-2}}$"
             f"  (2002 points, 280-4000 nm)",
             transform=axa.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    wl_b = (wl >= 300) & (wl <= 700)
    wl_r = (wl >= 380) & (wl <= 1400)
    axb.semilogy(wl[wl_r], r[wl_r] * np.interp(wl[wl_r], wl_s, e_s),
                 color="#2b6cb0", lw=2.0, ls="-", label="$S(\\lambda)\\,R(\\lambda)$ burn-weighted")
    axb.semilogy(wl[wl_b], b[wl_b] * np.interp(wl[wl_b], wl_s, e_s),
                 color="#c05621", lw=2.0, ls="--", label="$S(\\lambda)\\,B(\\lambda)$ blue-weighted")
    axb.set_xlim(300.0, 1400.0)
    axb.set_ylim(1e-3, 5e1)
    axb.set_xlabel("Wavelength $\\lambda$ (nm)")
    axb.set_ylabel("Weighted irradiance (W m$^{-2}$ nm$^{-1}$)")
    axb.set_title("Weighted spectrum (after weighting) -- S(lambda) times B(lambda) / R(lambda)")
    axb.grid(True, which="both", alpha=0.25)
    axb.axvspan(435.0, 440.0, color="#2b6cb0", alpha=0.08,
                label="R peak band 435-440 nm")
    i_r = int(np.argmax(r[wl_r] * np.interp(wl[wl_r], wl_s, e_s)))
    axb.text(0.03, 0.93,
             f"Peak $S\\,R$ at {wl[wl_r][i_r]:.0f} nm (blue-violet hump) "
             f"$\\rightarrow$ k$_R$ = {_iec.spectral_weights()['kR']:.3f}, "
             f"k$_B$ = {_iec.spectral_weights()['kB']:.4f}",
             transform=axb.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    handles, labels = _dedupe_legend(axb)
    axb.legend(handles=handles, labels=labels, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# --- figure 14: run profile (timing / stage provenance) --------------------------
# Phase colours, kept out of the RG scale on purpose: these describe where the
# run spent its wall time, not the hazard level.
_PROFILE_PHASE_COLORS = {
    "config": "#868E96",
    "site": "#2F9E44",
    "solar": "#F08C00",
    "analysis": "#1971C2",
    "io": "#9C27B0",
    "figures": "#E03131",
}
_PROFILE_OTHER = "#ADB5BD"


def plot_run_profile(timer, path, case_label="", min_pct=0.1):
    """Where a case spent its wall time, which function did what, and how the
    one parallel region was scheduled.

    Four panels:

    (a) inclusive wall time per phase, as a single 100%-wide stacked bar;
    (b) exclusive (self) time per instrumented function on a log axis, annotated
        with `calls x mean ms`; stages below `min_pct` of the total collapse into
        one "other" bar so the chart stays readable - the CSV keeps every stage;
    (c) thread-lane chart of the sub-panel tasks of the first analysed
        (array, receptor) pair - the only parallel region in the pipeline;
    (d) the numeric outputs of each function that are NOT already written to a
        CSV or a PNG, with units, plus the count of stages that are timing-only
        because their result is already stored.
    """
    from solar_glare.profiling import PHASES, PHASE_ROOT

    df = timer.to_frame()
    pframe = timer.phase_frame()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no stages recorded", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        return

    total_wall = float(timer.total_wall_s)
    fig = plt.figure(figsize=(16.5, 11.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          hspace=0.42, wspace=0.30)

    # ---------------------------------------------------- (a) phase breakdown
    axa = fig.add_subplot(gs[0, 0])
    left = 0.0
    for r in pframe.itertuples():
        w = 100.0 * r.wall_s / pframe["wall_s"].sum()
        axa.barh([0], [w], left=left, height=0.42,
                 color=_PROFILE_PHASE_COLORS.get(r.phase, _PROFILE_OTHER),
                 edgecolor="white", linewidth=1.2,
                 label=f"{r.phase}  {r.wall_s:.2f} s ({w:.1f}%)")
        if w > 6.0:
            axa.text(left + w / 2.0, 0.0, f"{r.phase}\n{w:.1f}%", ha="center",
                     va="center", color="white", fontsize=9, fontweight="bold")
        left += w
    axa.set_xlim(0, 100)
    axa.set_ylim(-0.6, 0.6)
    axa.set_yticks([])
    axa.set_xlabel("share of instrumented wall time (%)")
    axa.set_title(f"(a) Phase breakdown - {case_label or 'case'}", fontsize=11,
                  fontweight="bold")
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
               fontsize=8, frameon=False)
    axa.text(0.0, 0.42,
             f"total {total_wall:.1f} s wall - instrumented "
             f"{pframe['wall_s'].sum():.1f} s",
             fontsize=8, color="#495057")

    # -------------------------------------------------- (b) per-function time
    axb = fig.add_subplot(gs[0, 1])
    plot_df = df[df["self_s"] > 0].copy().sort_values("self_s")
    total_self = float(plot_df["self_s"].sum())
    cut = total_self * min_pct / 100.0
    small = plot_df[plot_df["self_s"] < cut]
    shown = plot_df[plot_df["self_s"] >= cut]
    if not small.empty and len(small) > 1:
        other_s = float(small["self_s"].sum())
        other_calls = int(small["calls"].sum())
        others = small["does"].str.cat(sep=" ")
    else:
        shown = plot_df
        other_s = 0.0
        other_calls = 0
        others = ""
    labels, values, colors, notes = [], [], [], []
    for r in shown.itertuples():
        labels.append(r.stage)
        values.append(r.self_s)
        colors.append(_PROFILE_PHASE_COLORS.get(r.phase, _PROFILE_OTHER))
        notes.append(f"{r.calls:,} x {r.mean_ms:.3g} ms")
    if other_s > 0:
        labels.append(f"other ({len(small)} stages)")
        values.append(other_s)
        colors.append(_PROFILE_OTHER)
        notes.append(f"{other_calls:,} calls")
    floor = max(total_self * 2e-4, 1e-6)
    ypos = np.arange(len(labels))
    axb.barh(ypos, [max(v, floor) for v in values], height=0.68, color=colors)
    axb.set_yticks(ypos)
    axb.set_yticklabels(labels, fontsize=8.5, family="monospace")
    axb.set_xscale("log")
    axb.set_xlim(floor, max(values) * 40.0)
    axb.set_xlabel("exclusive (self) wall time [s] - log scale")
    axb.set_title("(b) Time per function", fontsize=11, fontweight="bold")
    axb.grid(True, axis="x", which="both", alpha=0.25)
    # Stages inside the parallel region are measured on 16 threads at once, so
    # their self time is SUMMED over threads: the bar can exceed the total wall
    # time of the case. That is intended (it is total work), but it must be
    # stated or the chart looks wrong next to panel (a).
    axb.text(0.99, 0.02, "stages in the parallel region are summed\nover threads "
                         "(total work, may exceed wall time)",
             transform=axb.transAxes, ha="right", va="bottom", fontsize=7.5,
             color="#495057",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    for i, (v, nt) in enumerate(zip(values, notes)):
        axb.text(max(v, floor) * 1.35, i, f"{v:.3g} s   ({nt})", va="center",
                 fontsize=7.5, color="#343A40")
    handles = [Line2D([0], [0], color=c, lw=7,
                      label=p) for p, c in _PROFILE_PHASE_COLORS.items()
               if p in set(shown["phase"])]
    if other_s > 0:
        handles.append(Line2D([0], [0], color=_PROFILE_OTHER, lw=7, label="other"))
    axb.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

    # --------------------------------------------------- (c) parallel lanes
    axc = fig.add_subplot(gs[1, 0])
    spans = timer.span_frame()
    pair = df[df["stage"] == "analyze_pair"]
    par_tasks = int(pair["parallel_tasks"].iloc[0]) if len(pair) and \
        pd.notna(pair["parallel_tasks"].iloc[0]) else 0
    par_workers = int(pair["parallel_workers"].iloc[0]) if len(pair) and \
        pd.notna(pair["parallel_workers"].iloc[0]) else 0
    speed_txt = ""
    if len(pair) and pair["key_outputs"].iloc[0]:
        m = [s for s in pair["key_outputs"].iloc[0].split("; ")
             if s.startswith("speedup_vs_serial=")]
        if m:
            speed_txt = m[0].split("=", 1)[1]
    if not spans.empty and par_tasks:
        first = spans.sort_values("t0").head(par_tasks)
        origin = float(first["t0"].min())
        tids = list(dict.fromkeys(first["thread"].tolist()))
        cmap = plt.get_cmap("viridis")
        for lane, tid in enumerate(tids):
            rows = first[first["thread"] == tid]
            for r in rows.itertuples():
                axc.barh(lane, (r.t1 - r.t0) * 1e3, left=(r.t0 - origin) * 1e3,
                         height=0.62, color=cmap(r.index / max(par_tasks - 1, 1)),
                         edgecolor="white", linewidth=0.4)
        axc.set_yticks(range(len(tids)))
        axc.set_yticklabels([f"thread {i}" for i in range(len(tids))], fontsize=7.5)
        axc.set_xlabel("elapsed within the pair [ms]")
        axc.set_title(f"(c) Parallel region: {par_tasks} sub-panel tasks on "
                      f"{len(tids)} threads", fontsize=11, fontweight="bold")
        axc.grid(True, axis="x", alpha=0.25)
        axc.text(0.99, 0.02,
                 f"tasks {par_tasks} | peak concurrency {par_workers} threads"
                 + (f" | speedup {speed_txt}" if speed_txt else ""),
                 transform=axc.transAxes, ha="right", va="bottom", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray",
                           alpha=0.92))
    else:
        axc.text(0.5, 0.5, "no parallel region recorded", ha="center",
                 va="center", transform=axc.transAxes, fontsize=12, color="#C92A2A")
        axc.text(0.5, 0.38, "panel (c) has no data - spans were not captured",
                 ha="center", va="center", transform=axc.transAxes, fontsize=9,
                 color="#C92A2A")
        axc.axis("off")
        raise RuntimeError(
            "plot_run_profile: the parallel region reported "
            f"{par_tasks} task(s) but no task spans were recorded, so the "
            "lane chart would be empty. Pass a StageTimer with "
            "record_spans=True.")

    # ------------------------------------------------ (d) key-outputs table
    axd = fig.add_subplot(gs[1, 1])
    axd.axis("off")
    kept = df[df["key_outputs"].str.len() > 0]
    stored = df[(df["key_outputs"].str.len() == 0) & (df["persisted_in"].str.len() > 0)]
    head = f"{'function':<26} {'key output (value unit)':<62} self_s"
    lines = [head, "-" * len(head)]
    for r in kept.itertuples():
        lines.append(f"{r.stage:<26} {r.key_outputs[:62]:<62} {r.self_s:>7.3f}")
    if not kept.empty:
        lines.append("-" * len(head))
    for r in kept.itertuples():
        lines.append(f"  {r.does[:100]}")
    lines.append("")
    lines.append(f"{len(stored)} further stages are timing-only: their result is")
    lines.append("already written to a file, so the log does not repeat it:")
    for r in stored.head(6).itertuples():
        lines.append(f"  {r.stage:<26} -> {r.persisted_in[:56]}")
    if len(stored) > 6:
        lines.append(f"  ... and {len(stored) - 6} more (see run_profile.csv)")
    axd.text(0.0, 1.0, "\n".join(lines), transform=axd.transAxes, va="top",
             ha="left", fontsize=7.2, family="monospace", linespacing=1.45)
    axd.set_title("(d) Key outputs not already stored", fontsize=11,
                  fontweight="bold")

    fig.suptitle(f"Run profile - {case_label or 'case'} "
                 f"(timing + stage provenance; no physics recomputed)",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
