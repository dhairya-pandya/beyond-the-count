#!/usr/bin/env python
"""Render report/technical_report.md to report/technical_report.pdf (python-markdown + headless Google Chrome).

Optional convenience; needs `pip install markdown` and a local Chrome/Chromium (set CHROME=/path/to/binary to override).
"""
import os
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "report" / "technical_report.md"  # override: python scripts/16_build_report_pdf.py docs/PROBLEM_STATEMENT.md
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1d1d1f; }
h1 { font-size: 20pt; line-height: 1.2; margin-bottom: 4pt; } h2 { font-size: 15pt; margin-top: 20pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 14pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 8.8pt; width: 100%; page-break-inside: avoid; } th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; } th { background: #f0f2f5; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f4f4f6; padding: 0 2pt; } img { max-width: 100%; display: block; margin: 8pt auto; page-break-inside: avoid; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 10pt; color: #555; }
"""


def chrome() -> str:
    for c in (os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"):
        if c and Path(c).exists():
            return c
    sys.exit("Chrome/Chromium not found; set CHROME=/path/to/binary")


def main() -> None:
    global MD
    if len(sys.argv) > 1:
        MD = (ROOT / sys.argv[1]).resolve()
    HTML = MD.with_name("_" + MD.stem + ".html")
    PDF = MD.with_suffix(".pdf")
    body = markdown.markdown(MD.read_text(), extensions=["tables", "fenced_code", "sane_lists"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Cell Painting Shortcut Audit</title><style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True)
    HTML.unlink()
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
