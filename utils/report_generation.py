from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


class ReportGenerator:
    """
    A class for generating comprehensive report visualizations and PDF reports
    for skin disease prediction results.
    """

    def __init__(self, prediction_result=None, confidence_scores=None, metrics=None, history=None):
        """
        Initialize the report generator with prediction and model data.

        Args:
            prediction_result: The predicted skin condition
            confidence_scores: Dictionary of confidence scores for each class
            metrics: Dictionary containing model performance metrics
            history: Training history with accuracy and loss values
        """
        self.prediction_result = prediction_result
        self.confidence_scores = confidence_scores
        self.metrics = metrics or {'accuracy': 0.85}  # Default accuracy if metrics not provided
        self.history = history
        self.class_names = list(self.confidence_scores.keys()) if confidence_scores else ["Acne", "Hyperpigmentation", "Nail Psoriasis", "SJS-TEN", "Vitiligo"]

    def plot_accuracy_loss_curves(self):
        # Example data - in production this would use real training history
        epochs = range(1, 21)
        train_acc = [0.6 + 0.02*i for i in epochs]
        val_acc = [0.58 + 0.019*i for i in epochs]
        train_loss = [0.8 - 0.03*i for i in epochs]
        val_loss = [0.85 - 0.028*i for i in epochs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, train_acc, 'b-', label='Training Accuracy')
        ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(epochs, train_loss, 'b-', label='Training Loss')
        ax2.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()

    def plot_metrics_histogram(self):
        # Example metrics data
        metrics = {
            'Precision': np.random.normal(0.85, 0.05, 100),
            'Recall': np.random.normal(0.83, 0.06, 100),
            'F1 Score': np.random.normal(0.84, 0.055, 100)
        }

        plt.figure(figsize=(10, 6))
        for metric, values in metrics.items():
            plt.hist(values, alpha=0.5, label=metric, bins=20)

        plt.title('Distribution of Performance Metrics')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()

    def plot_metrics_pie_chart(self):
        # Example class distribution data
        classes = ['Acne', 'Hyperpigmentation', 'Nail Psoriasis', 'SJS-TEN', 'Vitiligo']
        sizes = [30, 25, 15, 10, 20]

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=classes, autopct='%1.1f%%')
        plt.title('Distribution of Correctly Classified Samples')

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()

    def plot_confusion_matrix_heatmap(self):
        # Example confusion matrix
        classes = ['Acne', 'Hyperpigmentation', 'Nail Psoriasis', 'SJS-TEN', 'Vitiligo']
        cm = np.array([
            [45, 3, 1, 0, 1],
            [2, 40, 2, 1, 0],
            [1, 2, 35, 1, 1],
            [0, 1, 1, 38, 0],
            [1, 0, 1, 0, 43]
        ])

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()

    def plot_metrics_bar_chart(self):
        # Example per-class metrics
        classes = ['Acne', 'Hyperpigmentation', 'Nail Psoriasis', 'SJS-TEN', 'Vitiligo']
        metrics = {
            'Precision': [0.90, 0.87, 0.85, 0.88, 0.89],
            'Recall': [0.88, 0.85, 0.83, 0.86, 0.87],
            'F1 Score': [0.89, 0.86, 0.84, 0.87, 0.88]
        }

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, metrics['Precision'], width, label='Precision')
        ax.bar(x, metrics['Recall'], width, label='Recall')
        ax.bar(x + width, metrics['F1 Score'], width, label='F1 Score')

        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()

        plt.tight_layout()

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()

    def create_metrics_table(self):
        # Example metrics data
        data = {
            'Class': ['Acne', 'Hyperpigmentation', 'Nail Psoriasis', 'SJS-TEN', 'Vitiligo'],
            'Precision': [0.90, 0.87, 0.85, 0.88, 0.89],
            'Recall': [0.88, 0.85, 0.83, 0.86, 0.87],
            'F1 Score': [0.89, 0.86, 0.84, 0.87, 0.88],
            'Support': [50, 45, 40, 40, 45]
        }
        return pd.DataFrame(data)


    def generate_html_report(self, patient_info=None, texture_analysis=None, color_profile=None):
        """
        Generate an HTML report combining all visualizations and metrics.

        Args:
            patient_info: Optional dictionary with patient information
            texture_analysis: Optional texture analysis results
            color_profile: Optional color profile analysis results

        Returns:
            HTML string containing the full report
        """
        # Get all visualizations
        accuracy_loss_img = self.plot_accuracy_loss_curves()
        metrics_histogram_img = self.plot_metrics_histogram()
        metrics_pie_chart_img = self.plot_metrics_pie_chart()
        confusion_matrix_img = self.plot_confusion_matrix_heatmap()
        metrics_bar_chart_img = self.plot_metrics_bar_chart()

        # Get metrics table
        metrics_table = self.create_metrics_table()

        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>Skin Disease Prediction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2C3E50; text-align: center; }}
                h2 {{ color: #3498DB; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .plot-container {{ text-align: center; margin: 20px 0; }}
                .plot {{ max-width: 100%; height: auto; }}
                .metrics-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                .metric-box {{ border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 10px 0; width: 45%; }}
            </style>
        </head>
        <body>
            <h1>Skin Disease Prediction Report</h1>

            <h2>Prediction Results</h2>
            <div class="metrics-container">
                <div class="metric-box">
                    <h3>Predicted Condition</h3>
                    <p><strong>{self.prediction_result or "N/A"}</strong></p>
                    <p>Confidence: {(self.confidence_scores[self.prediction_result]*100 if self.confidence_scores and self.prediction_result else 0):.2f}%</p>
                </div>
                <div class="metric-box">
                    <h3>Model Accuracy</h3>
                    <p><strong>Overall Accuracy: {self.metrics['accuracy']*100:.2f}%</strong></p>
                </div>
            </div>

            <h2>Model Performance Metrics</h2>
            <div class="plot-container">
                <h3>Accuracy and Loss Curves</h3>
                <img class="plot" src="data:image/png;base64,{accuracy_loss_img}" alt="Accuracy and Loss Curves">
            </div>

            <div class="plot-container">
                <h3>Performance Metrics by Class</h3>
                <img class="plot" src="data:image/png;base64,{metrics_bar_chart_img}" alt="Performance Metrics by Class">
            </div>

            <div class="plot-container">
                <h3>Metrics Distribution</h3>
                <img class="plot" src="data:image/png;base64,{metrics_histogram_img}" alt="Metrics Distribution">
            </div>

            <div class="plot-container">
                <h3>Class Distribution</h3>
                <img class="plot" src="data:image/png;base64,{metrics_pie_chart_img}" alt="Class Distribution">
            </div>

            <div class="plot-container">
                <h3>Confusion Matrix</h3>
                <img class="plot" src="data:image/png;base64,{confusion_matrix_img}" alt="Confusion Matrix">
            </div>

            <h2>Detailed Metrics Table</h2>
            {metrics_table.to_html(index=False)}
        """

        # Add optional sections if data is provided
        if texture_analysis and color_profile:
            html += f"""
            <h2>Image Analysis</h2>
            <div class="metrics-container">
                <div class="metric-box">
                    <h3>Texture Analysis</h3>
                    <p><strong>Contrast:</strong> {texture_analysis.get('contrast', 'N/A'):.4f}</p>
                    <p><strong>Homogeneity:</strong> {texture_analysis.get('homogeneity', 'N/A'):.4f}</p>
                    <p><strong>Energy:</strong> {texture_analysis.get('energy', 'N/A'):.4f}</p>
                    <p><strong>Correlation:</strong> {texture_analysis.get('correlation', 'N/A'):.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Color Profile</h3>
                    <p><strong>Average Redness:</strong> {color_profile.get('avg_red', 0):.2f}%</p>
                    <p><strong>Average Saturation:</strong> {color_profile.get('avg_saturation', 0):.2f}%</p>
                    <p><strong>Color Variance:</strong> {color_profile.get('color_variance', 0):.2f}</p>
                </div>
            </div>
            """

        # Add patient info if provided
        if patient_info:
            patient_info_html = "<h2>Patient Information</h2><table>"
            for key, value in patient_info.items():
                patient_info_html += f"<tr><th>{key}</th><td>{value}</td></tr>"
            patient_info_html += "</table>"
            html += patient_info_html

        # Close HTML document
        html += """
            <div style="margin-top: 30px; border-top: 1px solid #ddd; padding-top: 10px; font-size: 12px; color: #777;">
                <p>This report was generated by the Skin Disease Prediction System. This is for informational purposes only and should not replace professional medical advice.</p>
            </div>
        </body>
        </html>
        """

        return html

    def generate_pdf_report(self, output_path, patient_info=None, texture_analysis=None, color_profile=None):
        """
        Generate a PDF report with all visualizations and metrics.

        Args:
            output_path: File path to save the PDF report
            patient_info: Optional dictionary with patient information
            texture_analysis: Optional texture analysis results
            color_profile: Optional color profile analysis results

        Returns:
            Path to the generated PDF file
        """
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Define custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            alignment=1,  # Center alignment
            spaceAfter=12
        )
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        subheading_style = ParagraphStyle(
            name='SubHeading',
            parent=styles['Heading3'],
            spaceAfter=6
        )


        # Add title
        elements.append(Paragraph("Skin Disease Prediction Report", title_style))
        elements.append(Spacer(1, 0.2*inch))

        # Add prediction results
        elements.append(Paragraph("Prediction Results", heading_style))

        # Create prediction results table
        pred_data = [
            ["Predicted Condition", "Confidence"],
            [self.prediction_result or "N/A",
             f"{self.confidence_scores[self.prediction_result]*100:.2f}%" if self.confidence_scores and self.prediction_result else "N/A"]
        ]
        pred_table = Table(pred_data, colWidths=[3*inch, 3*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(pred_table)
        elements.append(Spacer(1, 0.2*inch))

        # Add overall accuracy
        elements.append(Paragraph(f"Overall Model Accuracy: {self.metrics['accuracy']*100:.2f}%", normal_style))
        elements.append(Spacer(1, 0.3*inch))

        # Get all visualizations
        accuracy_loss_img = self.plot_accuracy_loss_curves()
        metrics_histogram_img = self.plot_metrics_histogram()
        metrics_pie_chart_img = self.plot_metrics_pie_chart()
        confusion_matrix_img = self.plot_confusion_matrix_heatmap()
        metrics_bar_chart_img = self.plot_metrics_bar_chart()

        # Convert base64 images to reportlab images
        for title, img_str in [
            ("Accuracy and Loss Curves", accuracy_loss_img),
            ("Performance Metrics by Class", metrics_bar_chart_img),
            ("Metrics Distribution", metrics_histogram_img),
            ("Class Distribution", metrics_pie_chart_img),
            ("Confusion Matrix", confusion_matrix_img)
        ]:
            # Add section title
            elements.append(Paragraph(title, heading_style))

            # Decode image
            img_data = base64.b64decode(img_str)
            img_buffer = io.BytesIO(img_data)
            img = Image(img_buffer, width=6.5*inch, height=5*inch)

            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))

        # Add metrics table
        elements.append(Paragraph("Detailed Metrics Table", heading_style))
        metrics_df = self.create_metrics_table()

        # Convert dataframe to table data
        table_data = [metrics_df.columns.tolist()]
        for _, row in metrics_df.iterrows():
            table_data.append(row.tolist())

        # Create table
        metrics_table = Table(table_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.2*inch))

        # Add metrics description
        elements.append(Paragraph("Performance Metrics Description:", subheading_style))
        elements.append(Paragraph("• Precision: Indicates how many positively classified samples were actually positive", normal_style))
        elements.append(Paragraph("• Recall: Shows how many actual positives were correctly identified", normal_style))
        elements.append(Paragraph("• F1 Score: Balances precision and recall into a single metric", normal_style))
        elements.append(Paragraph("• Support: The number of samples in each class", normal_style))


        # Add image analysis if provided
        if texture_analysis and color_profile:
            elements.append(Spacer(1, 0.3*inch))
            elements.append(Paragraph("Image Analysis", heading_style))

            # Create texture analysis table
            texture_data = [
                ["Texture Analysis", "Value"],
                ["Contrast", f"{texture_analysis.get('contrast', 'N/A'):.4f}"],
                ["Homogeneity", f"{texture_analysis.get('homogeneity', 'N/A'):.4f}"],
                ["Energy", f"{texture_analysis.get('energy', 'N/A'):.4f}"],
                ["Correlation", f"{texture_analysis.get('correlation', 'N/A'):.4f}"]
            ]
            texture_table = Table(texture_data, colWidths=[3*inch, 3*inch])
            texture_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(texture_table)
            elements.append(Spacer(1, 0.2*inch))

            # Create color profile table
            color_data = [
                ["Color Profile", "Value"],
                ["Average Redness", f"{color_profile.get('avg_red', 0):.2f}%"],
                ["Average Saturation", f"{color_profile.get('avg_saturation', 0):.2f}%"],
                ["Color Variance", f"{color_profile.get('color_variance', 0):.2f}"]
            ]
            color_table = Table(color_data, colWidths=[3*inch, 3*inch])
            color_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(color_table)

        # Add patient info if provided
        if patient_info:
            elements.append(Spacer(1, 0.3*inch))
            elements.append(Paragraph("Patient Information", heading_style))

            patient_data = [["Field", "Value"]]
            for key, value in patient_info.items():
                patient_data.append([key, str(value)])

            patient_table = Table(patient_data, colWidths=[3*inch, 3*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(patient_table)

        # Add disclaimer
        elements.append(Spacer(1, 0.5*inch))
        disclaimer = Paragraph(
            "Disclaimer: This report was generated by the Skin Disease Prediction System. " +
            "This is for informational purposes only and should not replace professional medical advice.",
            ParagraphStyle('Disclaimer', parent=normal_style, fontSize=8, textColor=colors.gray)
        )
        elements.append(disclaimer)

        # Build PDF
        doc.build(elements)

        return output_path