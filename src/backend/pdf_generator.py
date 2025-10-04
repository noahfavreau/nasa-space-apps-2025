# report.py
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import pandas as pd

# Output filename
OUTPUT = "simple_report.pdf"

# Example table data (use pandas or any data source)
df = pd.DataFrame(
    {
        "Item": ["Apples", "Bananas", "Cherries"],
        "Quantity": [10, 5, 12],
        "Price": [0.5, 0.3, 1.2],
    }
)

# Create document
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40
)
styles = getSampleStyleSheet()
story = []

# Heading
heading_style = styles["Heading1"]
story.append(Paragraph("Monthly Produce Report", heading_style))
story.append(Spacer(1, 12))

# Description
desc_style = ParagraphStyle("desc", parent=styles["Normal"], fontSize=11, leading=14)
description = "This report lists produce items, quantities on hand, and unit prices. Totals are shown in the table below."
story.append(Paragraph(description, desc_style))
story.append(Spacer(1, 12))

# Table: convert DataFrame to list-of-lists with header
table_data = [list(df.columns)] + df.astype(str).values.tolist()

# Optional: add totals row
total_qty = df["Quantity"].sum()
total_value = (df["Quantity"] * df["Price"]).sum()
table_data.append(["Totals", str(total_qty), f"{total_value:.2f}"])

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

# Build PDF
doc.build(story)
print(f"PDF generated: {OUTPUT}")
