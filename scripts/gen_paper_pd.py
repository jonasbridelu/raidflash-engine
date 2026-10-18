#!/usr/bin/env python3
"""A4 PDF of docs/PAPER-PD.md (DejaVu)."""
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
PAPERS = [
    (ROOT / "docs" / "PAPER-PD.md", ROOT / "docs" / "PAPER-PD.pdf"),
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def inline(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts = s.split("**")
    s = "".join(f"<b>{p}</b>" if n % 2 else p for n, p in enumerate(parts))
    ticks = s.split("`")
    return "".join(f"<font face='DejaVuMono' size='8'>{p}</font>" if n % 2 else p for n, p in enumerate(ticks))


def is_sep(cells: list[str]) -> bool:
    return all(set(c.replace(":", "")) <= set("- ") and c.strip("-: ") == "" for c in cells)


def build(src: Path, out: Path) -> None:
    pdfmetrics.registerFont(TTFont("DejaVu", FONT))
    pdfmetrics.registerFont(TTFont("DejaVuB", FONTB))
    pdfmetrics.registerFont(TTFont("DejaVuMono", FONTM))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("T", fontName="DejaVuB", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=6))
    styles.add(ParagraphStyle("H1", fontName="DejaVuB", fontSize=12, leading=16, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle("H2", fontName="DejaVuB", fontSize=10, leading=13, spaceBefore=4, spaceAfter=2))
    styles.add(ParagraphStyle("B", fontName="DejaVu", fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=2))
    styles.add(ParagraphStyle("Td", fontName="DejaVu", fontSize=7.5, leading=10))
    styles.add(
        ParagraphStyle(
            "C",
            fontName="DejaVuMono",
            fontSize=8,
            leading=11,
            backColor=colors.HexColor("#f6f6f6"),
            leftIndent=4,
            rightIndent=4,
        )
    )

    flow = []
    lines = src.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            flow.append(Paragraph(inline(line[2:].strip()), styles["T"]))
            flow.append(Spacer(1, 3 * mm))
        elif line.startswith("## "):
            flow.append(Paragraph(inline(line[3:].strip()), styles["H1"]))
        elif line.startswith("### "):
            flow.append(Paragraph(inline(line[4:].strip()), styles["H2"]))
        elif line.strip() == "---":
            flow.append(Spacer(1, 2 * mm))
        elif line.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            flow.append(Preformatted("\n".join(buf) or " ", styles["C"]))
        elif line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not is_sep(cells):
                    rows.append([Paragraph(inline(c), styles["Td"]) for c in cells])
                i += 1
            i -= 1
            if rows:
                usable = A4[0] - 36 * mm
                n = len(rows[0])
                t = Table(rows, colWidths=[usable / n] * n, hAlign="LEFT", repeatRows=1)
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                        ]
                    )
                )
                flow.append(t)
                flow.append(Spacer(1, 2 * mm))
        elif line.startswith("- "):
            flow.append(Paragraph("• " + inline(line[2:]), styles["B"]))
        elif line.strip() == "":
            flow.append(Spacer(1, 1.5 * mm))
        else:
            flow.append(Paragraph(inline(line), styles["B"]))
        i += 1

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.drawString(18 * mm, 12 * mm, "RAIDFLASH PD · Jonathan Briceño De Luise · 2026 · Apache-2.0")
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="RAIDFLASH PD",
        author="Jonathan Briceño De Luise",
    )
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


def main() -> None:
    for src, out in PAPERS:
        build(src, out)


if __name__ == "__main__":
    main()
