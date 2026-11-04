"""Normalize an extracted PDF-to-text file into house style for AI parsing.

House style (see references/_FORMAT_62471.txt): units with '/' and '^'
(W/(m^2*sr)), multiplication as '*', exponents disambiguated (10^4, t^0.25),
decimal commas -> dots, equation stacks collapsed to one ASCII line, Greek
and math symbol chars preserved, no BOM UTF-8 LF.

Rules are data-driven (RULES + EQ_STACKS below). Groups A-E, H are generic
conventions; group F (equation stacks) and G (figure-legend junk) are
document-specific and ship with the BS EN 62471 defaults -- edit the lists
or empty them for other documents.

Usage:
    python normalize_pdf_text.py <file.txt> [--strict] [--dry-run]

Default = loose: every rule reports its count but never asserts, so a re-run
on an already-normalized file is safe (counts drop to 0). --strict asserts
each stack matches exactly once (like the original normalize_62471.py).
--dry-run prints the report without writing. UTF-8 only; never via PowerShell.
"""
import re
import sys

DOT = "[\\u22c5\\u00b7]"

# category -> 0/None: unbounded, no assert; 1: must match exactly once (strict)
RULES = [
    # ---------- A. UNITS ----------
    ("A", r"W" + DOT + r"m-2" + DOT + r"sr-1" + DOT + r"nm-1", "W/(m^2*sr*nm)"),
    ("A", r"W" + DOT + r"m-2" + DOT + r"nm-1" + DOT + r"sr-1", "W/(m^2*nm*sr)"),
    ("A", r"W" + DOT + r"m-2" + DOT + r"sr-1", "W/(m^2*sr)"),
    ("A", r"J" + DOT + r"m-2" + DOT + r"sr-1", "J/(m^2*sr)"),
    ("A", r"W" + DOT + r"m-2" + DOT + r"nm-1", "W/(m^2*nm)"),
    ("A", r"W/m2", "W/m^2"),
    ("A", r"cd" + DOT + r"m-2", "cd/m^2"),
    ("A", r"J" + DOT + r"m-2", "J/m^2"),
    ("A", r"W" + DOT + r"m-2", "W/m^2"),
    ("A", r"mm2", "mm^2"),
    ("A", r"W/\(m\^2\*sr\) nm-1", "W/(m^2*sr) nm^-1"),
    ("A", r"\[X\]" + DOT + r"nm-1", "[X]*nm^-1"),
    # ---------- B. SUBSCRIPTS / SYMBOL NAMES ----------
    ("B", r"αeff", "α_eff"),
    ("B", r"αmin", "α_min"),
    ("B", r"αmax", "α_max"),
    # ---------- C. INLINE MATH ----------
    ("C", r"0,011" + DOT + r"√\(t/10\)", "0.011*sqrt(t/10)"),
    ("C", r"0,011" + DOT + r"√\(t/100\)", "0.011*sqrt(t/100)"),
    ("C", r"0,0011" + DOT + r"√t", "0.0011*sqrt(t)"),
    ("C", r"α_min" + DOT + r"t0,5", "α_min*t^0.5"),
    ("C", r"α_eff = α_min" + DOT + r"√\(t/0,25\)", "α_eff = α_min*sqrt(t/0.25)"),
    ("C", r"d = α_eff" + DOT + r"H", "d = α_eff*H"),
    ("C", r"0,2" + DOT + r"100,02\(1150-" + "λ" + r"\)", "0.2*10^(0.02*(1150-λ))"),
    ("C", r"10\[\(450-" + "λ" + r"\)/50\]", "10^((450-λ)/50)"),
    ("C", r"10\[\(700-" + "λ" + r"\)/500\]", "10^((700-λ)/500)"),
    ("C", r"50000/\(α" + DOT + r"t 0,25\)", "50000/(α*t^0.25)"),
    ("C", r"18000/t 0,75", "18000/t^0.75"),
    ("C", r"20000/t 0,75", "20000/t^0.75"),
    ("C", r"540" + DOT + r"1012", "540*10^12"),
    # ---------- D. AMBIGUOUS EXPONENTS ----------
    ("D", r"\(7/3\)2", "(7/3)^2"),
    ("D", r"10-4", "10^-4"),
    ("D", r"104 cd", "10^4 cd"),
    ("D", r"of 104\.", "of 10^4."),
    ("D", r"approximately 104\.", "approximately 10^4."),
    ("D", r"of 106\.", "of 10^6."),
    ("D", r"106 nm", "10^6 nm"),
    ("D", r"106/t", "10^6/t"),
    ("D", r"104 s", "10^4 s"),
    ("D", r"≤ 10-2 W/m\^2", "≤ 10^-2 W/m^2"),
    # ---------- E. DECIMAL COMMA -> DOT (numbers only) ----------
    ("E", r"(?<=\d),(?=\d)", "."),
    # ---------- G. FIGURE-LEGEND JUNK (document-specific) ----------
    ("G", r"\[Wm-2sr-1\]", "[W/(m^2*sr)]"),
    ("G", r"\[Wm-2\]", "[W/m^2]"),
    ("G", r"L R_amin", "L_R(a_min)"),
    ("G", r"L R_amax", "L_R(a_max)"),
    ("G", r"L IR_amin", "L_IR(a_min)"),
    ("G", r"L IR_amax", "L_IR(a_max)"),
    ("G", r"L B", "L_B"),
    ("G", r"E s", "E_s"),
    ("G", r"E UVA", "E_UVA"),
    ("G", r"E IR", "E_IR"),
    ("G", r"E B", "E_B"),
]


def stack_pattern(literal):
    toks = re.split(r"\s+", literal.strip())
    return r"\s*".join(re.escape(t) for t in toks)


# (name, literal, replacement) -- collapse a PDF letter-stack into one line
# Values below are the BS EN 62471 equations 3.x / 5.x / Table 5.4-5.5 / 4.x.
EQ_STACKS = [
    ("""A
Φ
E
d
d
v
v =""", "E_v = dΦ_v / dA", "eq3.1"),
    ("""A
E
d
dΦ
=""", "E = dΦ / dA", "eq3.2"),
    ("""Ω
⋅
⋅
=
d
cos
d
d
v
v
θ
A
Φ
L""", "L_v = dΦ_v / (cosθ * dA * dΩ)", "eq3.3"),
    ("""Ω
θ
Φ
d
cos
d
d
⋅
⋅
=
A
L""", "L = dΦ / (cosθ * dA * dΩ)", "eq3.4"),
    ("""∫
⋅
=
t
t
Q
0
d
Φ""", "Q = ∫ Φ * dt", "eq3.5"),
    ("""A
Q
H
d
d
=""", "H = dQ / dA", "eq3.6a"),
    ("""∫
⋅
=
t
t
E
H
∆
d""", "H = ∫ E * d(∆t)", "eq3.6b"),
    ("""λ
λ
λ
d
)
(
dX
X
=""", "X_λ = dX(λ) / dλ", "eq3.7"),
    ("""( )
λ
λ
Φ
λ
d
d
d
⋅
=
A
E""", "E_λ = dΦ(λ) / (dA * dλ)", "eq3.8"),
    ("""( )
λ
Ω
θ
λ
Φ
λ
d
d
cos
d
d
⋅
⋅
⋅
=
A
L""", "L_λ = dΦ(λ) / (cosθ * dA * dΩ * dλ)", "eq3.9"),
    ("""Ω
⋅
= L
E
,""", "E = L * Ω", "eq5.2"),
    ("""4
2
γ
π
Ω
⋅
=""", "Ω = π * γ^2 / 4", "eq5.3"),
    ("""2
2
2
4
4
r
F
L
L
E
⋅
⋅
⋅
=
⋅
⋅
=
π
γ
π""", "E = L * π * γ^2 / 4 = L * π * F^2 / (4 * r^2)", "eq5.4"),
    ("""∑
⋅
⋅
=
λ
λ
λ
∆
)
(
s
S
E
E""", "E_s = Σ E_λ(λ) * S_UV(λ) * ∆λ", "T5.4 actinic"),
    ("""∑
⋅
=
λ
λ ∆
UVA
E
E""", "E_UVA = Σ E_λ * ∆λ", "T5.4 UV-A"),
    ("""∑
⋅
⋅
=
λ
λ
λ
∆
)
(
B
B
E
E""", "E_B = Σ E_λ(λ) * B(λ) * ∆λ", "T5.4 blue"),
    ("""∑
⋅
=
λ
λ
∆
IR
E
E""", "E_IR = Σ E_λ * ∆λ", "T5.4 IR"),
    ("""∑
⋅
=
λ
λ
∆
H
E
E""", "E_H = Σ E_λ * ∆λ", "T5.4 skin"),
    ("""∑
⋅
⋅
=
λ
λ
λ
∆
)
(
B
B
L
L""", "L_B = Σ L_λ(λ) * B(λ) * ∆λ", "T5.5 blue"),
    ("""∑
⋅
=
⋅
λ
λ
λ
∆
)
(
R
R
L
L""", "L_R = Σ L_λ(λ) * R(λ) * ∆λ", "T5.5 thermal"),
    ("""∑
⋅
=
⋅
λ
λ
λ
∆
)
(
IR
R
L
L""", "L_IR = Σ L_λ(λ) * R(λ) * ∆λ", "T5.5 weak"),
    ("""( )
30
∆
∆
)
(
,
400
200
UV
s
≤
⋅
⋅
⋅
=
⋅
∑∑
t
t
S
t
E
t
E
λ
λ
λ
λ""", "ΣΣ E_λ(λ,t) * S_UV(λ) * ∆λ * ∆t ≤ 30 J/m^2", "eq4.1"),
    ("""s
max
30
E
t
=""", "t_max = 30 / E_s  [s]", "eq4.2"),
    ("""( )
10000
∆
∆
t,
400
315
UVA
≤
⋅
⋅
=
⋅
∑∑
t
t
E
t
E
λ
λ
λ""", "ΣΣ E_λ(λ,t) * ∆λ * ∆t ≤ 10000 J/m^2", "eq4.3a"),
    ("""10
UVA ≤
E""", "E_UVA ≤ 10 W/m^2", "eq4.3b"),
    ("""UVA
max
10000
E
t
≤""", "t_max ≤ 10000 / E_UVA  [s]", "eq4.4"),
    ("""∑∑
≤
⋅
⋅
⋅
=
⋅
700
300
6
B
10
∆
∆
)
(
)
,
(
t
t
B
t
L
t
L
λ
λ
λ
λ""", "ΣΣ L_λ(λ,t) * B(λ) * ∆λ * ∆t ≤ 10^6 J/(m^2*sr)", "eq4.5a"),
    ("""∑
≤
⋅
⋅
=
700
300
B
100
∆
)
(
λ
λ
λ B
L
L""", "Σ L_λ(λ) * B(λ) * ∆λ ≤ 100 W/(m^2*sr)", "eq4.5b"),
    ("""B
6
max
10
L
t
=""", "t_max = 10^6 / L_B  [s]", "eq4.6"),
    ("""∑∑
≤
⋅
⋅
⋅
=
⋅
700
300
B
100
∆
∆
)
(
)
,
(
t
t
B
t
E
t
E
λ
λ
λ
λ""", "ΣΣ E_λ(λ,t) * B(λ) * ∆λ * ∆t ≤ 100 J/m^2", "eq4.7a"),
    ("""∑
≤
⋅
⋅
=
700
300
B
1
∆
)
(
λ
λ
λ B
E
E""", "Σ E_λ(λ) * B(λ) * ∆λ ≤ 1 W/m^2", "eq4.7b"),
    ("""B
max
100
E
t
=""", "t_max = 100 / E_B  [s]", "eq4.8"),
    ("""25
,
0
1400
380
R
50000
∆
)
(
t
R
L
L
⋅
≤
⋅
= ∑
⋅
α
λ
λ
λ""", "Σ L_λ(λ) * R(λ) * ∆λ ≤ 50000 / (α * t^0.25) W/(m^2*sr)", "eq4.9"),
    ("""α
λ
λ
λ
6000
∆
)
(
1400
780
IR
≤
⋅
= ∑
⋅R
L
L""", "Σ L_λ(λ) * R(λ) * ∆λ ≤ 6000 / α W/(m^2*sr)", "eq4.10"),
    ("""∑
−
⋅
⋅
≤
=
3000
780
75
,
0
IR
18000
∆
t
E
E
λ
λ""", "Σ E_λ(λ) * ∆λ ≤ 18000 / t^0.75 W/m^2", "eq4.11a"),
    ("""∑
≤
⋅
=
3000
780
IR
100
∆λ
λ
E
E""", "Σ E_λ(λ) * ∆λ ≤ 100 W/m^2", "eq4.11b"),
    ("""( )
25
,
0
3000
380
H
20000
∆
∆
,
t
t
t
E
t
E
t
⋅
≤
⋅
⋅
=
⋅
∑∑
λ
λ
λ""", "ΣΣ E_λ(λ,t) * ∆λ * ∆t ≤ 20000 * t^0.25 J/m^2", "eq4.12"),
]


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    strict = "--strict" in argv
    dry = "--dry-run" in argv
    if len(args) < 1 or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv[1:] else 1
    src = args[0]

    with open(src, "r", encoding="utf-8", newline="\n") as f:
        txt = f.read()

    report = []

    def apply(pattern, repl):
        nonlocal txt
        new, n = re.subn(pattern, repl, txt)
        txt = new
        return n

    for cat, pat, repl in RULES:
        n = apply(pat, repl)
        report.append(f"[{cat}] {pat!r}: {n}")

    for literal, repl, name in EQ_STACKS:
        n = apply(stack_pattern(literal), repl)
        report.append(f"[F] {name}: {n}")
        if strict and n != 1:
            print(f"EXPECT MISMATCH {name}: got {n}, expect 1")
            for r in report:
                print(r)
            return 1

    n_dot = len(re.findall(r"[⋅·]", txt))
    report.append(f"[H] leftover product dots before blanket: {n_dot}")
    apply(r"[⋅·]", "*")

    leftover = {
        "\\u22c5/\\u00b7 dots": len(re.findall("[\\u22c5\\u00b7]", txt)),
        "'m-2'": txt.count("m-2"),
        "'nm-1'": txt.count("nm-1"),
        "'sr-1'": txt.count("sr-1"),
        "digit,digit": len(re.findall(r"(?<=\\d),(?=\\d)", txt)),
        "'104 s'": txt.count("104 s"),
        "'106 nm'": txt.count("106 nm"),
        "'t 0,'": txt.count("t 0,"),
    }

    for r in report:
        print(r)
    print("-- leftover check --")
    for k, v in leftover.items():
        print(f"  {k}: {v}")

    if dry:
        print("DRY-RUN: file NOT written")
        return 0

    try:
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
    except Exception:
        print("WRITE FAILED - file NOT modified")
        for r in report:
            print(r)
        raise

    print(f"wrote: {src}  ({len(txt)} chars)")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))