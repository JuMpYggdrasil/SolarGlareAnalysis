"""Convert output/report.md to a self-contained HTML report with embedded images."""

import base64
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "output", "report.md")
HTML = os.path.join(ROOT, "output", "report.html")


def embed(match):
    src = match.group(1)
    path = os.path.join(ROOT, "output", src)
    if not os.path.exists(path):
        return f"<em>[missing {src}]</em>"
    ext = os.path.splitext(src)[1].lstrip(".").lower()
    mime = "image/svg+xml" if ext == "svg" else "image/png"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return (f'<img src="data:{mime};base64,{b64}" '
            'style="max-width:100%;margin:6px 0;border:1px solid #ddd;border-radius:4px">')


def main():
    md = open(MD, encoding="utf-8").read()
    body = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", embed, md)
    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>Solar Glare Hazard Analysis Report</title>
<style>
body{{font-family:'Segoe UI',Tahoma,sans-serif;max-width:920px;margin:24px auto;padding:0 16px;line-height:1.65;color:#222}}
h1{{border-bottom:3px solid #aa0044;padding-bottom:8px}}
h2{{color:#333;border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:32px}}
table{{border-collapse:collapse;margin:12px 0;font-size:14px}}
th,td{{border:1px solid #999;padding:6px 10px;text-align:left}}
th{{background:#eee}}
img{{max-width:100%}}
em{{color:#666;font-size:13px}}
code{{background:#f4f4f4;padding:2px 5px;border-radius:3px}}
pre{{background:#f4f4f4;padding:10px;border-radius:4px;overflow-x:auto}}
blockquote{{border-left:4px solid #ccc;margin:12px 0;padding:4px 14px;color:#444}}
</style>
</head>
<body>
{body}
</body>
</html>"""
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {HTML} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
