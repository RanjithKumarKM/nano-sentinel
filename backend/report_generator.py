import math
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def _format_percentage(value):
    if value is None:
        return "Unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    return f"{number:.2f}%" if math.isfinite(number) else "Unavailable"


def _format_probability(value):
    if value is None:
        return "Unavailable"
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not math.isfinite(probability) or probability < 0 or probability > 1:
        return "Unavailable"
    return _format_percentage(probability * 100)


def generate_inspection_pdf(record, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, severity = record.get("model", {}), record.get("severity", {})
    decision, source = record.get("decision", {}), record.get("source", {})
    probabilities = model.get("probabilities", {})
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"],
                              alignment=TA_CENTER, textColor=colors.HexColor("#12304A")))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"],
                              fontSize=8, leading=10, textColor=colors.HexColor("#555555")))
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=18*mm,
                            leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm,
                            title=f"Nano Sentinel Inspection {record.get('inspection_id', '')}")
    story = [Paragraph("NANO SENTINEL", styles["ReportTitle"]),
             Paragraph("Wafer Inspection Report", styles["Heading2"]), Spacer(1, 5*mm)]
    rows = [
        ["Inspection ID", str(record.get("inspection_id", "N/A"))],
        ["Inspection time (UTC)", str(record.get("timestamp_utc", "N/A"))],
        ["Input filename", str(source.get("filename", "N/A"))],
        ["Model", f"{model.get('name', 'N/A')} {model.get('version', '')}"],
        ["Predicted class", str(model.get("predicted_class", "N/A"))],
        ["Confidence", _format_percentage(model.get("confidence_percent"))],
        ["Inspection status", str(decision.get(
            "inspection_status", "N/A")).replace("_", " ")],
        ["Total die pixels", str(severity.get("total_die_pixels", "N/A"))],
        ["Defective die pixels", str(
            severity.get("defective_die_pixels", "N/A"))],
        ["Defect ratio",
            _format_percentage(severity.get("defect_ratio_percent"))],
        ["Provisional severity", str(
            severity.get("severity_category", "N/A"))],
    ]
    table = Table(rows, colWidths=[48*mm, 112*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F5")),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CBD5DF")),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [table, Spacer(1, 6*mm), Paragraph("Inspection recommendation", styles["Heading3"]),
              Paragraph(str(decision.get("recommendation",
                        "Review result manually.")), styles["BodyText"]),
              Spacer(1, 4*mm), Paragraph("Class probabilities", styles["Heading3"])]
    prob_rows = [["Class", "Probability (%)"]]
    if isinstance(probabilities, dict):
        prob_rows += [[str(k), _format_probability(v)]
                      for k, v in probabilities.items()]
    if len(prob_rows) == 1:
        prob_rows.append(["Unavailable", "No probability data was stored"])
    prob_table = Table(prob_rows, colWidths=[80*mm, 80*mm], repeatRows=1)
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CBD5DF")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [prob_table, Spacer(1, 6*mm), Paragraph(
        "Important: model confidence does not guarantee correctness. Defect ratio and severity "
        "are provisional pixel-map heuristics, not validated manufacturing standards. A qualified "
        "engineer should review the wafer map before taking corrective action.", styles["SmallNote"])]
    doc.build(story)
    return str(output_path)
