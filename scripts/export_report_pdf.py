from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FIGURES = [
    ("分领域 probe sweep", "figures/probe_sweep_summary.png"),
    ("Capital probe layer curve", "figures/probe_capital_answer.png"),
    ("Layer 8 activation PCA", "figures/pca_capital_layer8.png"),
    ("Activation patching recovery", "figures/activation_patching_capital_recall.png"),
    ("Probe-direction steering", "figures/steering_capital_probe_layer8_probe_accuracy.png"),
    ("Probe-direction ablation", "figures/ablation_capital_probe_layer8_score_gap.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/final_report.md")
    parser.add_argument("--out", default="reports/final_report.pdf")
    return parser.parse_args()


def register_fonts() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("ProjectCJK", str(path)))
            return "ProjectCJK"
    return "Helvetica"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "")
    )


def make_styles(font_name: str):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ProjectTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ProjectH1",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=16,
            leading=22,
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ProjectH2",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=18,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ProjectBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.8,
            leading=15,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ProjectCode",
            parent=styles["Code"],
            fontName=font_name,
            fontSize=8.6,
            leading=12,
            backColor=colors.HexColor("#F3F4F6"),
            borderPadding=5,
            leftIndent=6,
            rightIndent=6,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ProjectCaption",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4B5563"),
            spaceBefore=6,
            spaceAfter=8,
        )
    )
    return styles


def flush_paragraph(buffer: list[str], story: list, styles) -> None:
    if not buffer:
        return
    text = " ".join(line.strip() for line in buffer if line.strip())
    if text:
        story.append(Paragraph(esc(text), styles["ProjectBody"]))
    buffer.clear()


def add_table(rows: list[list[str]], story: list, styles, font_name: str) -> None:
    clean_rows = [[Paragraph(esc(cell.strip()), styles["ProjectBody"]) for cell in row] for row in rows]
    table = Table(clean_rows, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 6))


def parse_markdown(markdown: str, styles, font_name: str) -> list:
    story: list = []
    paragraph: list[str] = []
    code: list[str] = []
    table_rows: list[list[str]] = []
    bullet_items: list[str] = []
    in_code = False

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            add_table(table_rows, story, styles, font_name)
            table_rows = []

    def flush_bullets() -> None:
        nonlocal bullet_items
        if bullet_items:
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(esc(item), styles["ProjectBody"])) for item in bullet_items],
                    bulletType="bullet",
                    start="circle",
                    leftIndent=18,
                )
            )
            story.append(Spacer(1, 4))
            bullet_items = []

    for raw in markdown.splitlines():
        line = raw.rstrip()

        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code), styles["ProjectCode"]))
                code = []
                in_code = False
            else:
                flush_paragraph(paragraph, story, styles)
                flush_table()
                flush_bullets()
                in_code = True
            continue

        if in_code:
            code.append(line)
            continue

        if not line.strip():
            flush_paragraph(paragraph, story, styles)
            flush_table()
            flush_bullets()
            continue

        if line.startswith("# "):
            flush_paragraph(paragraph, story, styles)
            flush_table()
            flush_bullets()
            story.append(Paragraph(esc(line[2:].strip()), styles["ProjectTitle"]))
            continue

        if line.startswith("## "):
            flush_paragraph(paragraph, story, styles)
            flush_table()
            flush_bullets()
            story.append(Paragraph(esc(line[3:].strip()), styles["ProjectH1"]))
            continue

        if line.startswith("### "):
            flush_paragraph(paragraph, story, styles)
            flush_table()
            flush_bullets()
            story.append(Paragraph(esc(line[4:].strip()), styles["ProjectH2"]))
            continue

        if line.startswith("|") and line.endswith("|"):
            flush_paragraph(paragraph, story, styles)
            flush_bullets()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
                continue
            table_rows.append(cells)
            continue

        if line.lstrip().startswith("- "):
            flush_paragraph(paragraph, story, styles)
            flush_table()
            bullet_items.append(line.lstrip()[2:].strip())
            continue

        number_match = re.match(r"^\d+\.\s+(.*)$", line)
        if number_match:
            flush_paragraph(paragraph, story, styles)
            flush_table()
            bullet_items.append(number_match.group(1).strip())
            continue

        paragraph.append(line)

    flush_paragraph(paragraph, story, styles)
    flush_table()
    flush_bullets()
    return story


def add_figure_appendix(story: list, styles, page_width: float) -> None:
    story.append(PageBreak())
    story.append(Paragraph("附录：关键可视化结果", styles["ProjectH1"]))
    for title, rel_path in FIGURES:
        path = Path(rel_path)
        if not path.exists():
            continue
        story.append(Paragraph(esc(title), styles["ProjectH2"]))
        img = Image(str(path))
        max_width = page_width - 2 * cm
        max_height = 12 * cm
        ratio = min(max_width / img.imageWidth, max_height / img.imageHeight)
        img.drawWidth = img.imageWidth * ratio
        img.drawHeight = img.imageHeight * ratio
        story.append(img)
        story.append(Paragraph(rel_path, styles["ProjectCaption"]))
        story.append(Spacer(1, 8))


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(2 * cm, 1.1 * cm, "Mechanistic Interpretability Project")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))
    canvas.restoreState()


def main() -> None:
    args = parse_args()
    font_name = register_fonts()
    styles = make_styles(font_name)
    input_path = Path(args.input)
    output_path = Path(args.out)
    markdown = input_path.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Truth Direction Mechanistic Interpretability Report",
        author="Mechanistic Interpretability Course Project",
    )
    story = parse_markdown(markdown, styles, font_name)
    add_figure_appendix(story, styles, doc.width)
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"Saved PDF report to {output_path}")


if __name__ == "__main__":
    main()
