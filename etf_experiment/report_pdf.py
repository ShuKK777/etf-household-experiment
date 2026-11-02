#!/usr/bin/env python3
"""Render the report Markdown to a print-ready PDF with figures inlined.

WeasyPrint does the page layout. Two things need handling that plain Markdown does
not cover:

- **Images** are inlined as base64 rather than referenced by path, so the PDF builds
  correctly regardless of the working directory and the HTML can be inspected alone.
- **Numeric table cells** are right-aligned by a regex over the rendered HTML. A
  left-aligned column of signed percentage points is hard to scan, and Markdown has
  no column-level alignment hook that survives the table extension.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown
import weasyprint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "报告" / "实验汇总.md"
OUT = ROOT / "报告" / "实验汇总.pdf"

CSS = r"""
@page {
  size: A4;
  margin: 20mm 16mm 18mm 16mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: "Noto Sans CJK JP", sans-serif;
    font-size: 8pt;
    color: #888;
  }
  @top-right {
    content: "存款为什么不进 ETF · 家庭产品条件对照实验";
    font-family: "Noto Sans CJK JP", sans-serif;
    font-size: 7.5pt;
    color: #aaa;
  }
}
@page :first {
  @top-right { content: none; }
  @bottom-center { content: none; }
}

html { font-size: 10pt; }
body {
  font-family: "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Serif CJK JP", sans-serif;
  line-height: 1.62;
  color: #1a1a1a;
  text-align: justify;
  hyphens: none;
}

h1 {
  font-size: 21pt; line-height: 1.3; margin: 0 0 6pt 0;
  color: #14375e; letter-spacing: 0.5pt;
  padding-bottom: 8pt; border-bottom: 2.4pt solid #14375e;
  bookmark-level: 1;
}
h2 {
  font-size: 14pt; margin: 20pt 0 8pt 0; color: #14375e;
  padding-left: 7pt; border-left: 3.4pt solid #2f6b9e;
  page-break-after: avoid; page-break-before: auto;
  bookmark-level: 2;
}
h3 {
  font-size: 11.4pt; margin: 13pt 0 5pt 0; color: #1f4e79;
  page-break-after: avoid;
  bookmark-level: 3;
}
h4 {
  font-size: 10.4pt; margin: 10pt 0 4pt 0; color: #333;
  page-break-after: avoid; bookmark-level: 4;
}

p { margin: 0 0 6pt 0; orphans: 2; widows: 2; }
strong { color: #0d2b47; font-weight: 600; }
em { font-style: italic; }

a { color: #1a5c96; text-decoration: none; }

ul, ol { margin: 0 0 7pt 0; padding-left: 16pt; }
li { margin-bottom: 3pt; }
li > p { margin-bottom: 3pt; }

code {
  font-family: "Noto Sans Mono CJK JP", "DejaVu Sans Mono", monospace;
  font-size: 0.86em; background: #f2f4f7; color: #b03050;
  padding: 0.5pt 2.5pt; border-radius: 2pt;
}
pre {
  background: #f6f8fa; border-left: 2.5pt solid #b8c4d0;
  padding: 6pt 8pt; margin: 0 0 8pt 0; border-radius: 2pt;
  white-space: pre-wrap; word-wrap: break-word;
}
pre code { background: none; color: #24292e; padding: 0; font-size: 8pt; }

blockquote {
  margin: 8pt 0; padding: 6pt 10pt;
  background: #f4f8fc; border-left: 3pt solid #7ba7cf;
  color: #24384d;
}
blockquote p { margin: 0 0 3pt 0; }
blockquote p:last-child { margin-bottom: 0; }

table {
  width: 100%; border-collapse: collapse;
  margin: 4pt 0 10pt 0; font-size: 8.2pt;
  page-break-inside: auto;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; page-break-after: auto; }
th {
  background: #e8eef5; color: #14375e; font-weight: 600;
  padding: 3.5pt 4pt; border-bottom: 1.2pt solid #9db4ca;
  text-align: left; vertical-align: bottom;
}
td {
  padding: 3pt 4pt; border-bottom: 0.4pt solid #dde3ea;
  vertical-align: top;
}
tbody tr:nth-child(even) { background: #fafbfd; }
/* numeric-looking cells: right align */
td.num, th.num { text-align: right; }

img {
  display: block; max-width: 100%; height: auto;
  margin: 8pt auto 4pt auto;
  border: 0.4pt solid #d5dbe2; border-radius: 3pt;
  page-break-inside: avoid;
}

hr {
  border: none; border-top: 0.6pt solid #d5dbe2;
  margin: 14pt 0;
}

/* the generated doc uses "---" between intro blocks right after headings */
h2 + hr, h3 + hr, h1 + hr { display: none; }
"""


def embed_images(html: str, base_dir: Path) -> str:
    """Inline local images so weasyprint never needs a base_url guess."""

    def repl(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2)
        path = (base_dir / src).resolve()
        if not path.exists():
            print(f"!! missing image {src}")
            return match.group(0)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" alt="{alt}" />'

    return re.sub(r'<img alt="([^"]*)" src="([^"]*)"\s*/?>', repl, html)


def numeric_cells(html: str) -> str:
    """Right-align cells that look like numbers (incl. signed, %, pp, ranges)."""
    number = re.compile(
        r"^\s*(?:[−\-+±]?\d[\d,\.]*\s*(?:pp|%|元|户|次)?(?:\s*[~－-]\s*[−\-+]?\d[\d,\.]*\s*(?:pp|%|元)?)?"
        r"|[−\-+±]?\d[\d,\.]*\s*/\s*\d[\d,\.]*"
        r"|[0-9a-f]{6,})\s*$"
    )

    def fix_row(match: re.Match[str]) -> str:
        row = match.group(0)
        if "<th" in row:
            return row

        def cell(match_cell: re.Match[str]) -> str:
            body = match_cell.group(2)
            plain = re.sub(r"<[^>]+>", "", body).replace("**", "").strip()
            if plain and number.match(plain):
                return f'<td class="num">{body}</td>'
            return match_cell.group(0)

        return re.sub(r"<td([^>]*)>(.*?)</td>", cell, row, flags=re.S)

    return re.sub(r"<tr[^>]*>.*?</tr>", fix_row, html, flags=re.S)


def render(source: Path, output: Path) -> int:
    """Build the PDF from ``source`` and return the page count."""
    text = source.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    body = embed_images(body, source.parent)
    body = numeric_cells(body)
    html = (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<title>存款为什么不进 ETF：家庭产品条件对照实验</title>'
        '<meta name="author" content="ETF 家庭决策实验">'
        '<meta name="description" content="1 万户家庭画像 × 四条专用决策模型的产品条件对照实验汇总">'
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = weasyprint.HTML(string=html, base_url=str(source.parent)).render()
    doc.write_pdf(output)
    return len(doc.pages)


def main() -> None:
    pages = render(SRC, OUT)
    print(f"{OUT}  pages={pages}  {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
