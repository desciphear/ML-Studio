from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

def create_pdf_report(results, df_shape, model_name):
    """Create a PDF report of the model results without visualizations"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=20
    )
    story.append(Paragraph("Machine Learning Model Report", title_style))
    story.append(Spacer(1, 10))

    # Model Information Table
    model_info = [
        ["Model Type:", model_name],
        ["Dataset Shape:", f"{df_shape[0]} rows × {df_shape[1]} columns"],
        ["Model Accuracy:", f"{results['accuracy']:.2%}"]
    ]
    
    table = Table(model_info, colWidths=[120, 300])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('FONTSIZE', (0, 0), (-1, -1), 10)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Classification Report
    story.append(Paragraph("Classification Report", styles['Heading2']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<pre>{results['classification_report']}</pre>", styles['Code']))
    story.append(Spacer(1, 20))

    if model_name != "KNN":
        # Feature Importance Table
        story.append(Paragraph("Top Important Features", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        feature_importance = sorted(
            results['feature_importance'].items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]
        
        feature_data = [["Feature", "Importance"]] + [
            [f"{k}", f"{v:.4f}"] for k, v in feature_importance
        ]
        
        feature_table = Table(feature_data, colWidths=[300, 120])
        feature_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10)
        ]))
        story.append(feature_table)

    # Generate PDF
    doc.build(story)
    buffer.seek(0)
    return buffer