"""Compare ARC refractive index effects on E903 solar-weighted reflectivity

Plots R(AOI) for different ARC refractive indices (n=1.2, 1.3, 1.4, 1.5)
vs the full 5-layer PV module stack.
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
OUT_PATH = os.path.join(OUT_DIR, "pv_stack_arc_compare.txt")
FIG_PATH = os.path.join(OUT_DIR, "pv_stack_arc_compare.png")


def line(txt=""):
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(txt + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("ARC refractive index comparison vs full PV stack\n")
        f.write("Model: E903 smooth_arc with varying n_arc\n")
        f.write("=" * 70 + "\n\n")

    print("\n[A] Table at key AOIs")
    line("Glare-band AOIs (78-82°): smooth_arc with varying n vs full stack")
    line("")
    line("AOI   n=1.2      n=1.3      n=1.4      n=1.5      full_stack")
    line("-" * 65)

    aois_key = [78, 79, 80, 81, 82]
    for a in aois_key:
        r_n12 = float(hazard.solar_weighted_reflectivity(a, glass="smooth_arc", n_arc=1.2))
        r_n13 = float(hazard.solar_weighted_reflectivity(a, glass="smooth_arc", n_arc=1.3))
        r_n14 = float(hazard.solar_weighted_reflectivity(a, glass="smooth_arc", n_arc=1.4))
        r_n15 = float(hazard.solar_weighted_reflectivity(a, glass="smooth_arc", n_arc=1.5))
        r_st = float(hazard.pv_stack_reflectivity(a))
        line(f"{a:4.0f}  {r_n12:9.4f}  {r_n13:9.4f}  {r_n14:9.4f}  {r_n15:9.4f}  {r_st:12.4f}")

    print("\n[B] Plotting figure")
    aoi = np.linspace(0.0, 90.0, 901)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(aoi, hazard.solar_weighted_reflectivity(aoi, glass="smooth"),
            color="k", ls="-", lw=1.8, label="E903 smooth (bare glass)")
    ax.plot(aoi, hazard.solar_weighted_reflectivity(aoi, glass="smooth_arc", n_arc=1.2),
            color="tab:blue", ls="--", lw=1.4, label="E903 smooth_arc n=1.2")
    ax.plot(aoi, hazard.solar_weighted_reflectivity(aoi, glass="smooth_arc", n_arc=1.3),
            color="tab:red", ls="--", lw=1.8, label="E903 smooth_arc n=1.3 (default)")
    ax.plot(aoi, hazard.solar_weighted_reflectivity(aoi, glass="smooth_arc", n_arc=1.4),
            color="tab:green", ls="--", lw=1.4, label="E903 smooth_arc n=1.4")
    ax.plot(aoi, hazard.solar_weighted_reflectivity(aoi, glass="smooth_arc", n_arc=1.5),
            color="tab:orange", ls="--", lw=1.4, label="E903 smooth_arc n=1.5")
    ax.plot(aoi, hazard.pv_stack_reflectivity(aoi),
            color="tab:purple", ls="-", lw=2.0, label="Full PV stack (5-layer)")

    ax.set_xlabel("Angle of incidence θ (°)")
    ax.set_ylabel("Solar-weighted reflectance R")
    ax.set_title("Effect of ARC refractive index (n_arc) on reflectivity vs full PV stack")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)
    line("")
    line(f"Figure: {os.path.relpath(FIG_PATH, ROOT)}")

    print("\nDone. Text:", os.path.relpath(OUT_PATH, ROOT))
    print("Figure:", os.path.relpath(FIG_PATH, ROOT))


if __name__ == "__main__":
    main()
