"""Validate full PV module stack reflectivity vs current simplified models.

Compares the 5-layer PV module stack (from references/_PV_module_structure.txt)
with the current E903 solar-weighted models (smooth, smooth_arc). Uses constant
refractive indices as specified; no pass/fail gate.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solar_glare import hazard

OUT_DIR = os.path.join(ROOT, "output", "validation")
OUT_PATH = os.path.join(OUT_DIR, "pv_stack.txt")
FIG_PATH = os.path.join(OUT_DIR, "pv_stack_compare.png")


def line(txt=""):
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(txt + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("Full PV module stack reflectivity comparison\n")
        f.write("Source: references/_PV_module_structure.txt (5-layer)\n")
        f.write("=" * 70 + "\n\n")

    print("\n[A] Solar-weighted reflectance at key AOIs")
    line("Solar-weighted reflectance R(AOI) (unpolarised)")
    line("")
    line("AOI(deg)  smooth      smooth_arc  pv_stack_5L  Δ vs smooth_arc(%)")
    line("-" * 65)

    aois = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 82, 85, 88, 89, 90]
    for a in aois:
        r_s = float(hazard.solar_weighted_reflectivity(a, glass="smooth"))
        r_sa = float(hazard.solar_weighted_reflectivity(a, glass="smooth_arc"))
        r_st = float(hazard.pv_stack_reflectivity(a, use_pore_arc=True,
                                                  use_cell_arc=True))
        d_sa = 100.0 * (r_st - r_sa) / r_sa if r_sa > 1e-6 else 0.0
        line(f"{a:7.1f}  {r_s:10.6f}  {r_sa:10.6f}  {r_st:12.6f}  {d_sa:+9.2f}")

    line("")
    line("")
    line("[B] Glare-band AOIs (78-82°)")
    line("AOI(deg)  smooth_arc  pv_stack_5L  Δ(%)")
    line("-" * 45)
    for a in [78, 79, 80, 81, 82]:
        r_sa = float(hazard.solar_weighted_reflectivity(a, glass="smooth_arc"))
        r_st = float(hazard.pv_stack_reflectivity(a, use_pore_arc=True,
                                                  use_cell_arc=True))
        d_sa = 100.0 * (r_st - r_sa) / r_sa if r_sa > 1e-6 else 0.0
        line(f"{a:7.1f}  {r_sa:10.6f}  {r_st:12.6f}  {d_sa:+9.2f}")

    print("\n[C] Writing comparison figure", FIG_PATH)
    aoi = np.linspace(0.0, 90.0, 901)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(aoi, hazard.solar_weighted_reflectivity(aoi, glass="smooth"),
            color="k", ls="-", lw=1.6, label="E903 - smooth")
    ax.plot(aoi, hazard.solar_weighted_reflectivity(aoi, glass="smooth_arc"),
            color="k", ls="--", lw=1.6, label="E903 - smooth_arc")
    ax.plot(aoi, hazard.pv_stack_reflectivity(aoi, use_pore_arc=True,
                                              use_cell_arc=True),
            color="tab:blue", ls="-", lw=1.8, label="Full PV stack (5-layer)")
    ax.set_xlabel("Angle of incidence θ (deg)")
    ax.set_ylabel("Solar-weighted reflectance R")
    ax.set_title("Panel reflectivity: E903 simplified vs full 5-layer PV module stack")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)
    line("")
    line(f"\nComparison figure: {os.path.relpath(FIG_PATH, ROOT)}")

    print("\nDone. Output:", os.path.relpath(OUT_PATH, ROOT))
    print("Figure:", os.path.relpath(FIG_PATH, ROOT))


if __name__ == "__main__":
    main()
