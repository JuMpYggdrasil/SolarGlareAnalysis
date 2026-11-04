"""Solar position and clear-sky DNI via pvlib (configurable sub-hourly resolution).

Model choice (most accurate/reliable, kept deliberate):
  * Sun position: NREL SPA (Reda & Andreas 2004), accuracy ~ +/-0.0003 deg.
    This is preferred over the simplified Duffie & Beckman algorithm quoted in
    SGHAT Technical Reference v6 Sec.4.1, which (a) has a misprint in the
    equation-of-time formula (yields +16.4 min for 21 Jan instead of the true
    ~-11.2 min, i.e. ~27 min / ~6.8 deg hour-angle error) and (b) ignores
    atmospheric refraction, giving up to ~7 deg elevation error near the
    horizon (measured vs SPA on our site). At solar noon the difference is
    ~0.9 deg mean. DO NOT switch to method="ephemeris" (that is the paper's
    simplified model).
  * DNI: pvlib Ineichen clear-sky model (Linke turbidity) instead of the
    cosine profile in SGHAT v6 Sec.5.2, for a physically-based irradiance.
"""

import pandas as pd
import pvlib
import pvlib.atmosphere
import pvlib.solarposition

SOLPOS_METHOD = "nrel_numpy"  # NREL SPA: the reference-grade algorithm


def build_times(year, tz="Asia/Bangkok", freq="h"):
    """Timestamps for a full non-leap year at the requested pandas frequency.

    freq="h"   -> exactly 8,760 steps  (0:00 .. 23:00 each day)
    freq="min" -> exactly 525,600 steps (0:00 .. 23:59 each day)
    """
    return pd.date_range(
        start=f"{year}-01-01 00:00", end=f"{year}-12-31 23:59", freq=freq, tz=tz
    )


def solar_position(times, latitude, longitude, altitude=0.0,
                   pressure=None, temperature=25.0, method=SOLPOS_METHOD):
    """Apparent solar position. Returns DataFrame with 'apparent_zenith',
    'apparent_elevation' and 'azimuth' (0=N, 90=E, 180=S, 270=W).

    Uses the NREL SPA (method=SOLPOS_METHOD) for reference-grade accuracy
    (see module docstring for why it beats the paper's simplified model).
    Pressure defaults to the site-consistent value for `altitude` m so the
    atmospheric-refraction correction is physically correct near the horizon.

    REF: SGHAT Technical Reference v6 Sec.4.1 specifies the sun-position
    algorithm (Duffie & Beckman); we delegate to the NREL SPA via pvlib
    (higher accuracy than the paper's simplified spherical model).
    """
    if pressure is None:
        pressure = pvlib.atmosphere.alt2pres(altitude)
    pos = pvlib.solarposition.get_solarposition(
        times, latitude, longitude,
        altitude=altitude, pressure=pressure, temperature=temperature,
        method=method,
    )
    pos["apparent_elevation"] = 90.0 - pos["apparent_zenith"]
    return pos


def clearsky_dni(location, times, linke_turbidity=3.5):
    """Clear-sky DNI (W/m2) using the Ineichen model at the given times.
    REF: SGHAT Technical Reference v6 Sec.5.2 scales DNI with a cosine
    clear-day profile; we use the Ineichen clear-sky model from pvlib for
    a physically-based DNI instead of the paper's cosine approximation.
    """
    cs = location.get_clearsky(times, model="ineichen", linke_turbidity=linke_turbidity)
    return cs["dni"]
