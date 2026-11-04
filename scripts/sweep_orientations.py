"""Parameter sweep over panel orientation to find orientations that produce
glare at the configured receptors. Run from repo root:
    .venv\\Scripts\\python.exe scripts\\sweep_orientations.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from solar_glare.analysis import load_config, build_site, compute_ephemeris, analyze_pair
from solar_glare.geometry import enu_from_latlon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    cfg = load_config(os.path.join(ROOT, "config", "site.yaml"))
    site, arrays, receptors = build_site(cfg)
    times, pos, dni = compute_ephemeris(cfg, site)

    # template array (only position matters here; orientation is swept)
    base = arrays[0]
    tilts = [0, 5, 10, 15, 20, 30, 45, 60]
    azims = list(range(0, 360, 30))

    rows = []
    for a in arrays:
        import types
        for tilt in tilts:
            for az in azims:
                sweep = types.SimpleNamespace(
                    name=a.name, center_enu=a.center_enu,
                    char_size=a.char_size, specular_reflectivity=a.specular_reflectivity,
                    normal=None,
                )
                from solar_glare.geometry import surface_normal
                sweep.normal = surface_normal(tilt, az)
                total_g = total_y = total_r = 0
                for r in receptors:
                    codes, _ = analyze_pair(sweep, r, pos, dni)
                    total_g += int((codes == "G").sum())
                    total_y += int((codes == "Y").sum())
                    total_r += int((codes == "R").sum())
                if total_g + total_y + total_r > 0:
                    rows.append({
                        "array": a.name.split("(")[0].strip(),
                        "tilt": tilt, "azimuth": az,
                        "G": total_g, "Y": total_y, "R": total_r,
                    })
    df = pd.DataFrame(rows).sort_values(["Y", "G", "R"], ascending=False)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))
    print("\nTotal sweep rows with glare:", len(df))


if __name__ == "__main__":
    main()
