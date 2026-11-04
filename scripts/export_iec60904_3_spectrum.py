"""Export IEC 60904-3:2019 Table 1 (BS EN IEC 60904-3:2019) to a CSV.

Source of truth: the house-style text extract
    references/_BS_EN_IEC_60904-3-2019.txt
(= === PAGE N === markers, UTF-8 no BOM, normalized units).

The PDF text stack prints Table 1 as a VERTICAL token stream: each wavelength
row has 7 values (lambda, global, global photon flux, cumulative global,
direct, direct photon flux, cumulative direct) printed one-per-line, and the
table is interrupted every page by the repeated column header block, the
document title lines, and the === PAGE N === marker. Dur: skip every line that
does not parse as a float, group the surviving tokens in 7s, stop when the
wavelength token reaches 4000.0. The trailing "infinity / 900" closeout row
(summary of the cumulative integrals, printed as 1000/900) is NOT part of the
per-wavelength table and must be rejected: we simply stop after the 4000.0 row.

Output: src/solar_glare/data/IEC60904-3_2019_Table1.csv, 2002 data rows, all
seven Table 1 columns, comma-delimited with a header line.

Usage (from repo root):
    .venv\\Scripts\\python.exe scripts\\export_iec60904_3_spectrum.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_TXT = os.path.join(REPO, "references", "_BS_EN_IEC_60904-3-2019.txt")
OUT_CSV = os.path.join(REPO, "src", "solar_glare", "data",
                       "IEC60904-3_2019_Table1.csv")

COLS = [
    "wavelength_nm", "global_wm2nm", "global_photon_flux_1_m2_s_nm",
    "cumulative_global_wm2", "direct_wm2nm",
    "direct_photon_flux_1_m2_s_nm", "cumulative_direct_wm2",
]


def parse_table(txt_path):
    """Parse the vertical Table 1 token stream -> (2002, 7) float array.

    Collects every line that parses as a float from the first wavelength
    (280.0) onwards, groups the tokens in 7s, then takes the contiguous
    strictly-increasing prefix ending at 4000.0. The trailing infinity /
    1000 / 900 closeout row plus the Figure-1 axis values (1.8, 1.6, ...) and
    the SMARTS Table A.1 numbers regroup into bogus rows whose wavelength
    stops increasing; taking the monotonic prefix rejects all of them.
    """
    tokens = []
    took = False
    with open(txt_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                val = float(line.strip())
            except ValueError:
                continue  # header page, marker, blank, or the infinity row
            if val == 280.0 and not took:
                took = True  # first wavelength of Table 1
            if took:
                tokens.append(val)
    arr = np.array(tokens, dtype=np.float64)
    arr = arr[: arr.size // 7 * 7]  # drop trailing incomplete row
    rows = arr.reshape(-1, 7)
    inc = np.concatenate(([True], np.diff(rows[:, 0]) > 0))
    # first index where monotonicity breaks = length of the clean table prefix
    for i in range(1, len(rows)):
        if not (rows[i, 0] > rows[i - 1, 0] and abs(rows[i, 0] - rows[i - 1, 0]) < 60.0):
            rows = rows[:i]
            break
    return rows


def main():
    rows = parse_table(SRC_TXT)
    n = rows.shape[0]
    wl = rows[:, 0]
    print(f"parsed {n} rows, wavelength {wl[0]:.1f} .. {wl[-1]:.1f} nm")
    if n != 2002:
        raise SystemExit(f"expected 2002 rows, got {n}")
    if wl[0] != 280.0 or wl[-1] != 4000.0:
        raise SystemExit("wavelength range wrong")
    if not np.all(np.diff(wl) > 0):
        raise SystemExit("wavelength not monotonic")
    cum_d = rows[-1, 6]
    print(f"cumulative direct integrated irradiance at 4000 nm = "
          f"{cum_d:.2f} W/m2 (Table 1 closeout annotates 900)")
    if not (890.0 < cum_d < 905.0):
        raise SystemExit("cumulative direct integral out of expected range")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    np.savetxt(OUT_CSV, rows, delimiter=",", fmt="%.6e",
               header=",".join(COLS), comments="")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()