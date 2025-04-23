from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from io import BytesIO
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

def create_report(
    patient_name,
    prediction_result,
    confidence_scores,
    uploaded_image_pil,
    texture_analysis,
    color_profile,
    confidence_plot_base64=None,
    grad_cam_base64=None
):
    """
    Create a PDF report with analysis results.
    
    Args:
        patient_name: Name of the patient
        prediction_result: Predicted disease class
        confidence_scores: Dictionary of class labels and confidence scores
        uploaded_image_pil: PIL Image object of the uploaded image
        texture_analysis: Dictionary of texture analysis results
        color_profile: Dictionary of color profile analysis results
        confidence_plot_base64: Base64 encoded string of confidence scores plot
        grad_cam_base64: Base64 encoded string of Grad-CAM visualization
        
    Returns:
        PDF report as BytesIO object
    """
    # Create a BytesIO object to save the PDF
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Create a list of flowables to add to the document
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Add title
    elements.append(Paragraph(f"Skin Disease Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Add date and patient name
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Paragraph(f"Patient: {patient_name}", normal_style))
    elements.append(Spacer(1, 12))
    
    # Add prediction result
    elements.append(Paragraph("Prediction Result", heading_style))
    elements.append(Paragraph(f"Diagnosis: <b>{prediction_result}</b>", normal_style))
    elements.append(Spacer(1, 12))
    
    # Convert confidence scores to a table
    elements.append(Paragraph("Confidence Scores", subheading_style))
    
    # Create table data
    table_data = [["Disease", "Confidence"]]
    for disease, score in confidence_scores.items():
        table_data.append([disease, f"{score*100:.2f}%"])
    
    # Create table
    table = Table(table_data, colWidths=[2.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 12))
    
    # Add the uploaded image
    if uploaded_image_pil:
        # Resize image to fit on the page
        img_width, img_height = uploaded_image_pil.size
        aspect_ratio = img_height / float(img_width)
        new_width = 3 * inch
        new_height = new_width * aspect_ratio
        
        # Save image to BytesIO
        img_buffer = BytesIO()
        uploaded_image_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        elements.append(Paragraph("Uploaded Image", subheading_style))
        elements.append(Image(img_buffer, width=new_width, height=new_height))
        elements.append(Spacer(1, 12))
    
    # Add Grad-CAM visualization if available
    if grad_cam_base64:
        elements.append(Paragraph("Region Activation Map (Grad-CAM)", subheading_style))
        
        # Decode base64 image
        grad_cam_data = base64.b64decode(grad_cam_base64)
        grad_cam_buffer = BytesIO(grad_cam_data)
        
        # Add image to the document
        elements.append(Image(grad_cam_buffer, width=3*inch, height=3*inch))
        elements.append(Spacer(1, 12))
    
    # Add confidence plot if available
    if confidence_plot_base64:
        elements.append(Paragraph("Confidence Scores Visualization", subheading_style))
        
        # Decode base64 image
        plot_data = base64.b64decode(confidence_plot_base64)
        plot_buffer = BytesIO(plot_data)
        
        # Add image to the document
        elements.append(Image(plot_buffer, width=4*inch, height=3*inch))
        elements.append(Spacer(1, 12))
    
    # Add texture analysis results
    elements.append(Paragraph("Texture Analysis", heading_style))
    texture_text = ""
    for key, value in texture_analysis.items():
        texture_text += f"<b>{key.capitalize()}:</b> {value:.4f}<br/>"
    elements.append(Paragraph(texture_text, normal_style))
    elements.append(Spacer(1, 12))
    
    # Add color profile results
    elements.append(Paragraph("Color Profile Analysis", heading_style))
    
    # RGB Stats
    elements.append(Paragraph("RGB Statistics", subheading_style))
    rgb_table_data = [["Channel", "Mean", "Standard Deviation"]]
    rgb_table_data.append(["Red", f"{color_profile['rgb']['r_mean']:.2f}", f"{color_profile['rgb']['r_std']:.2f}"])
    rgb_table_data.append(["Green", f"{color_profile['rgb']['g_mean']:.2f}", f"{color_profile['rgb']['g_std']:.2f}"])
    rgb_table_data.append(["Blue", f"{color_profile['rgb']['b_mean']:.2f}", f"{color_profile['rgb']['b_std']:.2f}"])
    
    rgb_table = Table(rgb_table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    rgb_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(rgb_table)
    elements.append(Spacer(1, 12))
    
    # HSV Stats
    elements.append(Paragraph("HSV Statistics", subheading_style))
    hsv_table_data = [["Channel", "Mean", "Standard Deviation"]]
    hsv_table_data.append(["Hue", f"{color_profile['hsv']['h_mean']:.2f}", f"{color_profile['hsv']['h_std']:.2f}"])
    hsv_table_data.append(["Saturation", f"{color_profile['hsv']['s_mean']:.2f}", f"{color_profile['hsv']['s_std']:.2f}"])
    hsv_table_data.append(["Value", f"{color_profile['hsv']['v_mean']:.2f}", f"{color_profile['hsv']['v_std']:.2f}"])
    
    hsv_table = Table(hsv_table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    hsv_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(hsv_table)
    elements.append(Spacer(1, 12))
    
    # Disclaimer
    elements.append(Paragraph("Disclaimer", heading_style))
    disclaimer_text = (
        "This report is generated for informational purposes only and does not constitute "
        "medical advice. Please consult with a qualified healthcare professional for "
        "proper diagnosis and treatment. The predictions made by this system are based "
        "on machine learning algorithms and should be interpreted by medical professionals."
    )
    elements.append(Paragraph(disclaimer_text, normal_style))
    
    # Build the PDF document
    doc.build(elements)
    
    # Get the value of the BytesIO buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    return pdf_value
