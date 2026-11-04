"""3D geometry visualizations (matplotlib)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geometry import sun_vector, reflect, surface_normal
from .viz import _route_color


def _panel_corners(array):
    """Corner coordinates of a panel rectangle in its local ENU plane."""
    n = surface_normal(array.tilt_deg, array.azimuth_deg)
    # build two orthonormal in-plane vectors
    up = np.array([0.0, 0.0, 1.0])
    e = up - n * np.dot(up, n)          # project up onto plane
    if np.linalg.norm(e) < 1e-9:        # horizontal panel (normal == up)
        e = np.array([1.0, 0.0, 0.0])
    else:
        e = e / np.linalg.norm(e)
    s = np.cross(n, e)                  # in-plane, orthogonal to e
    c = array.center_enu
    w, h = array.width_m / 2.0, array.height_m / 2.0
    return np.array([
        c + e * w + s * h,
        c + e * w - s * h,
        c - e * w - s * h,
        c - e * w + s * h,
    ])


def plot_glare_geometry(array, receptor, sun_az_deg, sun_el_deg, path, title=None,
                        all_arrays=None, occurrence_time=None,
                        g_min=0, y_min=0, r_min=0):
    """3D view: panel(s), sun vector, reflected ray to receptor.

    `all_arrays` (optional): every PV array to draw as semi-transparent panels;
    the analyzed `array` is drawn opaque and its reflected ray (worst-case)
    is shown along with the receptor.

    `occurrence_time` (optional): local timestamp string of the shown worst
    minute; `g_min`/`y_min`/`r_min` (optional): annual glare minutes per
    hazard class. When given, they are drawn as a text box in a clear corner
    of the figure (not in the title).
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # other panels (all arrays except the analyzed one): translucent
    for other in (all_arrays or []):
        if other is array:
            continue
        oc = _panel_corners(other)
        ofc = np.vstack([oc, oc[0]])
        ax.plot(ofc[:, 0], ofc[:, 1], ofc[:, 2], "-", color="#a0a0a0", lw=1.0, alpha=0.7)
        ax.plot_trisurf(oc[:, 0], oc[:, 1], oc[:, 2],
                        color="#a0a0a0", alpha=0.18, shade=True)

    corners = _panel_corners(array)
    poly = np.vstack([corners, corners[0]])
    ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], "-", color="#ff851b", lw=1.5)
    ax.plot_trisurf(corners[:, 0], corners[:, 1], corners[:, 2],
                    color="#ff851b", alpha=0.35, shade=True)

    s = sun_vector(sun_az_deg, sun_el_deg)
    n = surface_normal(array.tilt_deg, array.azimuth_deg)
    r = reflect(s, n)
    sun_tip = array.center_enu + s * 1500.0
    ray_tip = array.center_enu + r * 1500.0

    ax.quiver(array.center_enu[0], array.center_enu[1], array.center_enu[2],
              s[0], s[1], s[2], length=600.0, color="#ffdc00", normalize=True,
              label="Sun direction")
    ax.quiver(array.center_enu[0], array.center_enu[1], array.center_enu[2],
              r[0], r[1], r[2], length=600.0, color="#ff4136", normalize=True,
              label="Reflected ray")
    ax.quiver(array.center_enu[0], array.center_enu[1], array.center_enu[2],
              n[0], n[1], n[2], length=400.0, color="gray", normalize=True,
              label="Panel normal")

    is_obs = receptor.name in {"obs_asia", "obs_ptt"}
    marker = "o" if is_obs else "^"
    ax.scatter(*receptor.enu, marker=marker, s=120, c="#3d9970",
               edgecolor=_route_color(receptor.name), linewidths=1.3,
               label="Observation point" if is_obs else "Receptor")
    ax.plot([array.center_enu[0], receptor.enu[0]],
            [array.center_enu[1], receptor.enu[1]],
            [array.center_enu[2], receptor.enu[2]], "--", color="#3d9970", alpha=0.5)

    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Up (m)")
    ax.set_title(title or f"{array.name} -> {receptor.name}\n"
                          f"sun az {sun_az_deg:.1f} deg, el {sun_el_deg:.1f} deg")
    ax.legend(loc="upper left", fontsize=8)
    lim = np.max([np.max(np.abs(corners)) + 100.0, 1800.0])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, lim)
    ax.view_init(elev=20, azim=-55)

    info_lines = []
    if occurrence_time:
        info_lines.append(f"Occurred (local): {occurrence_time}")
    if info_lines or g_min or y_min or r_min:
        info_lines.append(f"Accumulate per Year: R {r_min} / G {g_min} / Y {y_min} min")
        fig.text(0.5, 0.02, "\n".join(info_lines), transform=fig.transFigure,
                 fontsize=9, va="bottom", ha="center",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           alpha=0.92, edgecolor="0.6"))

    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
