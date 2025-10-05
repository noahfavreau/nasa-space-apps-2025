# report.py
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.graphics.charts.piecharts import Pie
import pandas as pd
from reportlab.graphics.shapes import Drawing
import os


def produce_pdf():
    # Output filename
    OUTPUT = "simple_report.pdf"

    # Example table data (use pandas or any data source)
    df = pd.DataFrame({"Item": [1, 2, 3], "Output": [5, 35, 50]})

    # Create document
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=60,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    story = []

    # Heading
    heading_style = styles["Title"]
    story.append(Paragraph("Exoplanet discovery sheet", heading_style))
    story.append(Spacer(1, 12))

    # Description
    desc_style = ParagraphStyle(
        "desc", parent=styles["Normal"], fontSize=11, leading=14
    )
    description = "This report lists produce items, quantities on hand, and unit prices. Totals are shown in the table below."
    story.append(Paragraph(description, desc_style))
    story.append(Spacer(1, 12))

    # Table: convert DataFrame to list-of-lists with header
    table_data = [list(df.columns)] + df.astype(str).values.tolist()
    total_qty = df["Output"].sum()

    for pos, i in enumerate(table_data):
        if pos == 0:
            table_data[pos].append(r"% of data")
            continue
        table_data[pos].append(f"{int(i[1]) / total_qty * 100:.2f}%")

    # Optional: add totals row
    table_data.append(["Totals", str(total_qty), f"{'100%'}"])

    # Build table
    table = Table(table_data, hAlign="LEFT", colWidths=[200, 80, 80])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, -1), (-1, -1), colors.whitesmoke),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(table)
    # plt.pie(
    #     exo_stats,
    #     labels=df["Item"],
    #     autopct="%1.1f%%",
    #     startangle=90,
    # )
    d = Drawing()
    d.width = 100
    d.height = 100

    pie = Pie(angleRange=360)
    pie.data = df["Output"].tolist()
    pie.labels = [str(i) for i in df["Item"].tolist()]
    pie.y = 100
    pie.x = 100
    pie.slices.strokeWidth = 0.5
    d.add(pie)
    story.append(d)
    # Build PDF
    story.append
    doc.build(story)
    print(f"PDF generated: {OUTPUT}")


produce_pdf()
