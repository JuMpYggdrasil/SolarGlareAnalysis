"""Extract PDF text -> UTF-8 (no BOM, LF) with === PAGE N === markers.

Usage:
    python pdf_extract.py <input.pdf> [output.txt]

Default output = same dir/stem as input with .txt. Page separators from
PyMuPDF page order. Read-only on the PDF; writes UTF-8 no-BOM, newline=\\n.
Run with the project venv interpreter; set PYTHONIOENCODING=utf-8 on this
machine's cp874 console when piping.
"""
import os
import re
import sys

try:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24 naming
    except ImportError:
        import fitz  # legacy alias
except ImportError as e:
    sys.exit(f"PyMuPDF not installed (add 'PyMuPDF' to requirements.txt): {e}")


def out_default(src: str) -> str:
    stem = re.sub(r"\.pdf$", "", src, flags=re.IGNORECASE)
    return stem + ".txt"


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(argv) > 1 else 1
    src = argv[1]
    dst = argv[2] if len(argv) > 2 else out_default(src)
    if not os.path.isfile(src):
        sys.exit(f"input not found: {src}")

    doc = fitz.open(src)
    parts = []
    for i, page in enumerate(doc, start=1):
        parts.append(f"=== PAGE {i} ===")
        parts.append(page.get_text("text"))

    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts))

    print(f"pages: {len(doc)}")
    print(f"chars: {sum(len(p) for p in parts)}")
    print(f"wrote: {os.path.abspath(dst)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))