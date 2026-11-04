"""Vector geometry helpers for solar glare ray tracing.

All vectors live in a local right-handed ENU frame:
  East = (1,0,0), North = (0,1,0), Up = (0,0,1).

Solar azimuth follows the pvlib convention: 0 = North, 90 = East,
180 = South, 270 = West (clockwise when viewed from above).
"""

import numpy as np

from .profiling import stage_fn

R_EARTH = 6371000.0  # mean earth radius (m); spherical approx, fine for the few-km site extents here


def enu_from_latlon(ref_lat, ref_lon, lat, lon, height):
    """Convert (lat, lon, height_m) to local ENU meters about a reference point.

    Spherical approximation, adequate for the few-km site extents here.
    """
    ref_lat_r = np.radians(ref_lat)
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    ref_lon_r = np.radians(ref_lon)
    x = R_EARTH * np.cos(ref_lat_r) * (lon_r - ref_lon_r)
    y = R_EARTH * (lat_r - ref_lat_r)
    return np.array([float(x), float(y), float(height)], dtype=float)


def enu_to_latlon(ref_lat, ref_lon, x, y):
    """Inverse of enu_from_latlon: convert local ENU (x=E, y=N, metres) to
    (lat, lon) about a reference point. Spherical approximation.
    """
    ref_lat_r = np.radians(ref_lat)
    ref_lon_r = np.radians(ref_lon)
    lat = ref_lat_r + y / R_EARTH
    lon = ref_lon_r + x / (R_EARTH * np.cos(ref_lat_r))
    return float(np.degrees(lat)), float(np.degrees(lon))


@stage_fn("sun_vector")
def sun_vector(azimuth_deg, elevation_deg):
    """Unit vector pointing FROM a point TOWARD the sun.

    Accepts scalars or arrays of shape (N,); returns (3,) or (N,3).
    REF: SGHAT Technical Reference v6 Sec.4.1 - components
    s_e = cos(alt)*sin(az), s_n = cos(alt)*cos(az), s_u = sin(alt).
    """
    az = np.radians(np.asarray(azimuth_deg, dtype=float))
    el = np.radians(np.asarray(elevation_deg, dtype=float))
    e = np.cos(el) * np.sin(az)
    n = np.cos(el) * np.cos(az)
    u = np.sin(el)
    if e.ndim == 0:
        return np.array([float(e), float(n), float(u)])
    return np.column_stack((e, n, u))


def surface_normal(tilt_deg, azimuth_deg):
    """Outward unit normal of a panel surface.

    tilt_deg: angle from horizontal (0 = flat, 90 = vertical).
    azimuth_deg: compass direction the surface faces (0 = N, 180 = S).
    REF: SGHAT Technical Reference v6 Sec.4.5 (fixed-tilt limit of the
    single-axis normal construction; tilt from horizontal, azimuth from N).
    """
    t = np.radians(tilt_deg)
    az = np.radians(azimuth_deg)
    e = np.sin(t) * np.sin(az)
    n = np.sin(t) * np.cos(az)
    u = np.cos(t)
    return np.array([float(e), float(n), float(u)])


@stage_fn("reflect")
def reflect(incident, normal):
    """Reflect incident unit vectors about a surface normal (mirror law).

    incident: (3,) or (N,3) unit vectors pointing toward the source (i.e. the
        vector from the surface toward the sun).
    normal: (3,) unit surface normal.
    Returns (3,) or (N,3) unit vectors giving the DIRECTION OF TRAVEL of the
    reflected light (from the surface toward the observer).
    REF: SGHAT Technical Reference v6 Sec.4.2 - reflected sun vector
    r = 2 (s.n) n - s. This is the reflection of the *light travel direction*
    (-s) about the surface normal: v_out = v_in - 2(v_in.n) n with v_in = -s.
    (Beware: some texts write r = s - 2(s.n) n, which is the NEGATIVE - the
    direction from the observer back toward the panel.)
    """
    inc = np.asarray(incident, dtype=float)
    scalar = inc.ndim == 1
    if scalar:
        inc = inc[None, :]
    inc = inc / np.linalg.norm(inc, axis=-1, keepdims=True)
    nor = np.asarray(normal, dtype=float)
    nor = nor / np.linalg.norm(nor)
    proj = inc @ nor
    out = 2.0 * proj[:, None] * nor[None, :] - inc
    return out[0] if scalar else out


@stage_fn("angle_between")
def angle_between(a, b):
    """Angle in radians between vectors a (3,) or (N,3) and b (3,) or (N,3)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scalar = a.ndim == 1
    if scalar:
        a = a[None, :]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    if b.ndim == 1:
        b = b / np.linalg.norm(b)
        dot = a @ b
    else:
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        dot = (a * b).sum(axis=-1)
    out = np.arccos(np.clip(dot, -1.0, 1.0))
    return out[0] if scalar else out


def angle_between_vec(a, b):
    """Vectorized angle between a (N,3) and b (M,3) or (3,).
    
    Returns (N,) or (N, M) angles in radians. Batch version of angle_between
    for pre-filter and edge distance calculations.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    dots = a_norm @ b_norm.T
    return np.arccos(np.clip(dots, -1.0, 1.0))


def _panel_angular_span(corners_enu, receptor_enu, u_hat):
    """Max angular deviation of 4 panel corners from center direction u_hat.
    
    Vectorized replacement for the 4-iteration Python loop in pre-filter.
    """
    d_c = corners_enu - receptor_enu
    d_c = d_c / np.linalg.norm(d_c, axis=1, keepdims=True)
    dots = (d_c * u_hat).sum(axis=1)
    return float(np.max(np.arccos(np.clip(dots, -1.0, 1.0))))


def panel_axes(normal):
    """Orthonormal axes in the panel plane: e = width axis, s = height axis."""
    up = np.array([0.0, 0.0, 1.0])
    n = np.asarray(normal, dtype=float)
    e = up - n * np.dot(up, n)
    if np.linalg.norm(e) < 1e-9:
        e = np.array([1.0, 0.0, 0.0])
    else:
        e = e / np.linalg.norm(e)
    s = np.cross(n, e)
    return e, s


@stage_fn("panel_corners_enu")
def panel_corners_enu(center_enu, normal, width_m, height_m):
    """4 corner positions (4,3) of a flat rectangular panel in ENU.

    Corners follow CCW winding: (+e+s), (-e+s), (-e-s), (+e-s).
    """
    e, s = panel_axes(normal)
    c = np.asarray(center_enu, dtype=float)
    hw, hh = width_m / 2.0, height_m / 2.0
    return np.array([
        c + hw * e + hh * s,
        c - hw * e + hh * s,
        c - hw * e - hh * s,
        c + hw * e - hh * s,
    ])


@stage_fn("glare_polygon_hit")
def glare_polygon_hit(s_vec, corners_enu, receptor_enu, normal, half_spread,
                      subset=None):
    """True-rectangle polygon hit test (polyhedral emanation).

    Projects the 4 panel corners onto the receiver's viewing sphere, reflects
    them about the panel normal to obtain the 4 specular sun directions v_i,
    then tests whether the actual sun direction s_vec lies inside the
    resulting spherical convex polygon dilated by half_spread (β/2).

    The spherical polygon is evaluated with exact great-circle geometry
    (edge plane half-spaces + angular distance to each edge arc), so it
    remains correct even when the panel subtends a large solid angle
    (elongated rows seen from close range) - the tangent-plane
    approximation would distort those cases.

    For a flat specular panel this is the analytic equivalent of sampling
    the full rectangle (SGHAT Sec.4.4), but O(1) in panel resolution.

    `subset` is an optional (N,) boolean mask restricting the test to those
    rows only (e.g. after a cheap conservative prefilter); the other rows are
    returned False.  This keeps the method O(N) exact at full-year scale even
    though the polygon test is ~50x costlier per row than the circular cone.

    s_vec:        (3,) or (N,3) unit sun vectors (toward the sun).
    corners_enu:  (4,3) panel corner positions in ENU.
    receptor_enu: (3,)  receptor position in ENU.
    normal:       (3,)  panel outward unit normal.
    half_spread:  beam half-angle β/2 (rad).
    subset:       optional boolean mask of length N.
    Returns:      (N,) boolean mask, or scalar bool if s_vec is (3,).
    """
    s = np.asarray(s_vec, dtype=float)
    scalar = s.ndim == 1
    if scalar:
        s = s[None, :]
    N = s.shape[0]

    if subset is not None:
        sub = np.asarray(subset, dtype=bool)
        if sub.shape[0] != N:
            raise ValueError("subset length must match s_vec")
        if not sub.any():
            return bool(False) if scalar else np.zeros(N, dtype=bool)
        s_batch = s[sub]
        hit_sub = _glare_polygon_hit_rows(s_batch, corners_enu, receptor_enu,
                                          normal, half_spread)
        full = np.zeros(N, dtype=bool)
        full[sub] = hit_sub
        return bool(full[0]) if scalar else full

    hit = _glare_polygon_hit_rows(s, corners_enu, receptor_enu, normal,
                                  half_spread)
    return bool(hit[0]) if scalar else hit


def _glare_polygon_hit_rows(s, corners_enu, receptor_enu, normal, half_spread):
    """Row-wise core of glare_polygon_hit (s is (N,3))."""
    N = s.shape[0]
    C = np.asarray(corners_enu, dtype=float)  # (4, 3)
    P = np.asarray(receptor_enu, dtype=float)  # (3,)

    # Direction from each panel corner to receptor (corner → receptor)
    d = P - C  # (4, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)

    # Specular sun directions: v_i = reflect(d_i, n) = 2(d_i·n)n − d_i
    n = np.asarray(normal, dtype=float)
    v = 2.0 * np.einsum('ij,j->i', d, n)[:, None] * n[None, :] - d  # (4, 3)

    centroid = v.sum(axis=0)
    cn = np.linalg.norm(centroid)
    if cn < 1e-9:
        # Degenerate (panel visible as ~a point): fall back to angular
        # distance to the mean specular direction.
        if not np.any(v.sum(axis=0)):
            cdir = v[0]
        else:
            cdir = v.sum(axis=0)
        return angle_between(s, cdir) <= half_spread
    cdir = centroid / cn

    inside = np.ones(N, dtype=bool)
    dist = np.full(N, np.inf)
    for i in range(4):
        A = v[i]                    # (3,)
        B = v[(i + 1) % 4]          # (3,)
        nc = np.cross(A, B)         # great-circle edge normal
        ncn = np.linalg.norm(nc)
        if ncn < 1e-12:             # edge degenerates -> drop
            continue
        nc = nc / ncn
        if nc @ cdir < 0.0:         # orient so centroid sits on the + side
            nc = -nc
        dp = s @ nc                             # signed distance (sin-space)
        inside &= dp >= 0.0

        # Angular distance from s to the edge arc (A-B):
        # perpendicular foot on the great circle if inside the arc span,
        # otherwise the nearer endpoint.
        dotAB = A @ B
        foot = s - (s @ nc)[:, None] * nc[None, :]   # (N,3) in GC plane
        fn = np.linalg.norm(foot, axis=1)
        ok = fn > 1e-12
        foot_n = np.zeros_like(foot)
        foot_n[ok] = foot[ok] / fn[ok, None]
        fA = np.where(ok, (foot_n @ A), np.inf)      # (N,)
        fB = np.where(ok, (foot_n @ B), np.inf)
        on_arc = (fA >= dotAB - 1e-9) & (fB >= dotAB - 1e-9)
        d_plane = np.abs(np.arcsin(np.clip(dp, -1.0, 1.0)))
        dA = np.arccos(np.clip(s @ A, -1.0, 1.0))
        dB = np.arccos(np.clip(s @ B, -1.0, 1.0))
        d_edge = np.where(on_arc, d_plane, np.minimum(dA, dB))
        dist = np.minimum(dist, d_edge)

    return inside | (dist <= half_spread)
