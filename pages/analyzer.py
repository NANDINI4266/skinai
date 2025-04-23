import streamlit as st
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from utils.image_processing import preprocess_image, analyze_texture, analyze_color_profile, extract_roi, plot_color_distribution
from utils.model_utils import load_model, predict_disease, generate_grad_cam, plot_confidence_scores
from utils.report_generator import create_report
from datetime import datetime

def show_analyzer():
    """
    Display the disease analyzer page.
    """
    st.title("Skin Disease Analyzer")
    st.markdown("---")
    
    # Create tabs for different analyzer functionalities
    tabs = st.tabs(["Image Upload", "Analysis Results", "Advanced Analysis", "Report Generation"])
    
    # Image Upload Tab
    with tabs[0]:
        st.subheader("Upload Skin Image")
        
        # Create two columns for upload options
        col1, col2 = st.columns(2)
        
        with col1:
            # Image upload widget
            uploaded_file = st.file_uploader(
                "Choose a skin image file", 
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of the skin condition for analysis."
            )
        
        with col2:
            st.markdown("""
            ### For Best Results:
            - Upload a clear, well-lit image
            - Ensure the condition is clearly visible
            - Avoid images with filters or enhancements
            - Image should be in focus with minimal background
            """)
        
        # Display uploaded image and perform analysis
        if uploaded_file is not None:
            # Load and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Save image to session state
            st.session_state.uploaded_image = image
            
            # Process image button
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Processing image..."):
                    # Preprocess image
                    preprocessed_image = preprocess_image(image)
                    
                    # Perform texture analysis
                    texture_analysis = analyze_texture(image)
                    st.session_state.texture_analysis = texture_analysis
                    
                    # Perform color profile analysis
                    color_profile = analyze_color_profile(image)
                    st.session_state.color_profile = color_profile
                    
                    # Load model and make prediction
                    model = load_model()
                    
                    # Perform real-time analysis and prediction on the image
                    try:
                        # Show a progress bar for the analysis
                        progress_bar = st.progress(0)
                        
                        # Stage 1: Initial preprocessing (25%)
                        progress_bar.progress(25)
                        
                        # Stage 2: Model prediction using real-time analysis (50%)
                        progress_bar.progress(50)
                        # Get the actual prediction using the real-time analysis
                        predicted_class, confidence_scores = predict_disease(image, model)
                        
                        # Save results to session state
                        st.session_state.prediction_result = predicted_class
                        st.session_state.confidence_scores = confidence_scores
                        
                        # Stage 3: Generate visualization (75%)
                        progress_bar.progress(75)
                        
                        # Generate confidence plot from real analysis results
                        confidence_plot = plot_confidence_scores(confidence_scores)
                        st.session_state.confidence_plot = confidence_plot
                        
                        # Generate Grad-CAM visualization using real-time analysis
                        grad_cam_base64 = generate_grad_cam(image, model)
                        st.session_state.grad_cam = grad_cam_base64
                        
                        # Import here to avoid circular imports
                        from utils.model_metrics import calculate_metrics_from_features
                        
                        # Generate model metrics based on image features
                        image_features = {
                            'texture': st.session_state.texture_analysis,
                            'color': st.session_state.color_profile
                        }
                        metrics = calculate_metrics_from_features(image_features)
                        st.session_state.model_metrics = metrics
                        
                        # Stage 4: Complete (100%)
                        progress_bar.progress(100)
                        
                        st.success(f"Analysis complete! Detected condition: {predicted_class}")
                        st.info("View detailed results in the 'Analysis Results' tab.")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    
    # Analysis Results Tab
    with tabs[1]:
        st.subheader("Analysis Results")
        
        # Check if prediction has been made
        if 'prediction_result' in st.session_state and st.session_state.prediction_result is not None:
            # Create columns for results display
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Display basic prediction results
                st.markdown("### Prediction")
                st.markdown(f"**Detected Condition:** {st.session_state.prediction_result}")
                
                # Display confidence for predicted class
                predicted_class = st.session_state.prediction_result
                confidence = st.session_state.confidence_scores[predicted_class]
                st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                
                # Display information about the detected condition
                st.markdown("### About this condition")
                
                condition_info = {
                    "Acne": """
                    A skin condition that occurs when hair follicles plug with oil and dead skin cells.
                    Common features include whiteheads, blackheads, pimples, and inflammatory lesions.
                    """,
                    "Hyperpigmentation": """
                    A condition where patches of skin become darker than surrounding areas due to excess melanin.
                    Can be caused by sun damage, inflammation, or other skin injuries.
                    """,
                    "Nail Psoriasis": """
                    A manifestation of psoriasis affecting the fingernails and toenails.
                    Features may include pitting, discoloration, and nail bed separation.
                    """,
                    "SJS-TEN": """
                    Stevens-Johnson syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) are severe skin reactions.
                    Usually caused by medications and characterized by extensive skin detachment and mucosal involvement.
                    """,
                    "Vitiligo": """
                    A condition where skin loses its pigment cells, resulting in white patches.
                    Caused by the destruction of melanocytes and can affect any part of the body.
                    """
                }
                
                st.info(condition_info.get(predicted_class, "Information not available for this condition."))
            
            with col2:
                # Show confidence chart if available
                if 'confidence_plot' in st.session_state and st.session_state.confidence_plot:
                    st.markdown("### Confidence Scores")
                    st.image(
                        "data:image/png;base64," + st.session_state.confidence_plot,
                        caption="Disease Prediction Confidence Scores",
                        use_container_width=True
                    )
                
                # Show Grad-CAM visualization if available
                if 'grad_cam' in st.session_state and st.session_state.grad_cam:
                    st.markdown("### Region Activation Map (Grad-CAM)")
                    st.image(
                        "data:image/png;base64," + st.session_state.grad_cam,
                        caption="Highlighted regions of interest for diagnosis",
                        use_container_width=True
                    )
        else:
            st.info("No analysis results yet. Please upload and analyze an image first.")
    
    # Advanced Analysis Tab
    with tabs[2]:
        st.subheader("Advanced Analysis")
        
        # Check if image has been uploaded
        if 'uploaded_image' in st.session_state and st.session_state.uploaded_image is not None:
            # Create tabs for different analysis types
            analysis_tabs = st.tabs(["Texture Analysis", "Color Profile", "Region of Interest"])
            
            # Texture Analysis Tab
            with analysis_tabs[0]:
                st.markdown("### Texture Analysis")
                
                if 'texture_analysis' in st.session_state and st.session_state.texture_analysis is not None:
                    # Display texture metrics
                    texture = st.session_state.texture_analysis
                    
                    # Create a nice display for texture metrics
                    metrics_cols = st.columns(5)
                    
                    with metrics_cols[0]:
                        st.metric("Contrast", f"{texture['contrast']:.4f}")
                    
                    with metrics_cols[1]:
                        st.metric("Dissimilarity", f"{texture['dissimilarity']:.4f}")
                    
                    with metrics_cols[2]:
                        st.metric("Homogeneity", f"{texture['homogeneity']:.4f}")
                    
                    with metrics_cols[3]:
                        st.metric("Energy", f"{texture['energy']:.4f}")
                    
                    with metrics_cols[4]:
                        st.metric("Correlation", f"{texture['correlation']:.4f}")
                    
                    # Texture interpretation
                    st.markdown("### Interpretation")
                    
                    # Provide some interpretation based on texture values
                    interpretations = []
                    
                    if texture['contrast'] > 0.5:
                        interpretations.append("- **High contrast**: Indicates significant texture variations which can be seen in conditions with varied surface patterns.")
                    else:
                        interpretations.append("- **Low contrast**: Suggests a more uniform texture which may be seen in conditions with smoother surfaces.")
                    
                    if texture['homogeneity'] > 0.5:
                        interpretations.append("- **High homogeneity**: Indicates more uniform regions which can suggest continuous lesions.")
                    else:
                        interpretations.append("- **Low homogeneity**: Suggests more varied texture which can be seen in conditions with scattered lesions.")
                    
                    if texture['energy'] > 0.2:
                        interpretations.append("- **High energy**: Indicates more uniform or ordered texture patterns.")
                    else:
                        interpretations.append("- **Low energy**: Suggests more random or disordered texture patterns.")
                    
                    st.markdown("\n".join(interpretations))
                    
                else:
                    st.info("Texture analysis not available. Please analyze the image first.")
            
            # Color Profile Tab
            with analysis_tabs[1]:
                st.markdown("### Color Profile Analysis")
                
                if 'color_profile' in st.session_state and st.session_state.color_profile is not None:
                    color_profile = st.session_state.color_profile
                    
                    # Display RGB metrics
                    st.markdown("#### RGB Analysis")
                    rgb_cols = st.columns(3)
                    
                    with rgb_cols[0]:
                        st.markdown("**Red Channel**")
                        st.metric("Mean", f"{color_profile['rgb']['r_mean']:.2f}")
                        st.metric("Std Dev", f"{color_profile['rgb']['r_std']:.2f}")
                    
                    with rgb_cols[1]:
                        st.markdown("**Green Channel**")
                        st.metric("Mean", f"{color_profile['rgb']['g_mean']:.2f}")
                        st.metric("Std Dev", f"{color_profile['rgb']['g_std']:.2f}")
                    
                    with rgb_cols[2]:
                        st.markdown("**Blue Channel**")
                        st.metric("Mean", f"{color_profile['rgb']['b_mean']:.2f}")
                        st.metric("Std Dev", f"{color_profile['rgb']['b_std']:.2f}")
                    
                    # Display HSV metrics
                    st.markdown("#### HSV Analysis")
                    hsv_cols = st.columns(3)
                    
                    with hsv_cols[0]:
                        st.markdown("**Hue**")
                        st.metric("Mean", f"{color_profile['hsv']['h_mean']:.2f}")
                        st.metric("Std Dev", f"{color_profile['hsv']['h_std']:.2f}")
                    
                    with hsv_cols[1]:
                        st.markdown("**Saturation**")
                        st.metric("Mean", f"{color_profile['hsv']['s_mean']:.2f}")
                        st.metric("Std Dev", f"{color_profile['hsv']['s_std']:.2f}")
                    
                    with hsv_cols[2]:
                        st.markdown("**Value (Brightness)**")
                        st.metric("Mean", f"{color_profile['hsv']['v_mean']:.2f}")
                        st.metric("Std Dev", f"{color_profile['hsv']['v_std']:.2f}")
                    
                    # Display LAB metrics
                    st.markdown("#### LAB Analysis")
                    lab_cols = st.columns(3)
                    
                    with lab_cols[0]:
                        st.markdown("**Lightness**")
                        st.metric("Mean", f"{color_profile['lab']['l_mean']:.2f}")
                        st.metric("Std Dev", f"{color_profile['lab']['l_std']:.2f}")
                    
                    with lab_cols[1]:
                        st.markdown("**A (Green-Red)**")
                        st.metric("Mean", f"{color_profile['lab']['a_mean']:.2f}")
                        st.metric("Std Dev", f"{color_profile['lab']['a_std']:.2f}")
                    
                    with lab_cols[2]:
                        st.markdown("**B (Blue-Yellow)**")
                        st.metric("Mean", f"{color_profile['lab']['b_mean']:.2f}")
                        st.metric("Std Dev", f"{color_profile['lab']['b_std']:.2f}")
                    
                    # Create color distribution plot
                    st.markdown("#### Color Distribution")
                    
                    # Generate color distribution plot
                    image = st.session_state.uploaded_image
                    color_dist_plot = plot_color_distribution(image)
                    
                    # Display the plot
                    st.image(
                        "data:image/png;base64," + color_dist_plot,
                        caption="Color Channel Distribution",
                        use_container_width=True
                    )
                    
                else:
                    st.info("Color profile analysis not available. Please analyze the image first.")
            
            # Region of Interest Tab
            with analysis_tabs[2]:
                st.markdown("### Region of Interest (ROI) Selection")
                
                # Display the image
                image = st.session_state.uploaded_image
                st.image(image, caption="Select a region of interest", use_container_width=True)
                
                # ROI selection controls
                st.markdown("#### Define Region of Interest")
                
                # Get image dimensions
                img_width, img_height = image.size
                
                # Create two columns for X and Y coordinates
                col1, col2 = st.columns(2)
                
                with col1:
                    x = st.slider("X Position", 0, img_width - 50, img_width // 4, help="Left position of ROI")
                    width = st.slider("Width", 50, img_width - x, min(200, img_width - x), help="Width of ROI")
                
                with col2:
                    y = st.slider("Y Position", 0, img_height - 50, img_height // 4, help="Top position of ROI")
                    height = st.slider("Height", 50, img_height - y, min(200, img_height - y), help="Height of ROI")
                
                # Extract and display ROI
                roi = extract_roi(image, x, y, width, height)
                st.session_state.roi_selection = roi
                
                # Display ROI
                st.image(roi, caption="Selected Region of Interest", width=300)
                
                # Analyze ROI button
                if st.button("Analyze ROI"):
                    with st.spinner("Analyzing ROI..."):
                        # Perform texture analysis on ROI
                        roi_texture = analyze_texture(roi)
                        
                        # Perform color analysis on ROI
                        roi_color = analyze_color_profile(roi)
                        
                        # Display results
                        st.markdown("#### ROI Analysis Results")
                        
                        # Texture metrics for ROI
                        st.markdown("**Texture Metrics**")
                        roi_metrics_cols = st.columns(5)
                        
                        with roi_metrics_cols[0]:
                            st.metric("Contrast", f"{roi_texture['contrast']:.4f}")
                        
                        with roi_metrics_cols[1]:
                            st.metric("Dissimilarity", f"{roi_texture['dissimilarity']:.4f}")
                        
                        with roi_metrics_cols[2]:
                            st.metric("Homogeneity", f"{roi_texture['homogeneity']:.4f}")
                        
                        with roi_metrics_cols[3]:
                            st.metric("Energy", f"{roi_texture['energy']:.4f}")
                        
                        with roi_metrics_cols[4]:
                            st.metric("Correlation", f"{roi_texture['correlation']:.4f}")
                        
                        # Color metrics for ROI
                        st.markdown("**Color Metrics**")
                        roi_color_cols = st.columns(3)
                        
                        with roi_color_cols[0]:
                            st.markdown("**RGB Means**")
                            st.metric("Red", f"{roi_color['rgb']['r_mean']:.2f}")
                            st.metric("Green", f"{roi_color['rgb']['g_mean']:.2f}")
                            st.metric("Blue", f"{roi_color['rgb']['b_mean']:.2f}")
                        
                        with roi_color_cols[1]:
                            st.markdown("**HSV Means**")
                            st.metric("Hue", f"{roi_color['hsv']['h_mean']:.2f}")
                            st.metric("Saturation", f"{roi_color['hsv']['s_mean']:.2f}")
                            st.metric("Value", f"{roi_color['hsv']['v_mean']:.2f}")
                        
                        with roi_color_cols[2]:
                            st.markdown("**LAB Means**")
                            st.metric("Lightness", f"{roi_color['lab']['l_mean']:.2f}")
                            st.metric("A (Green-Red)", f"{roi_color['lab']['a_mean']:.2f}")
                            st.metric("B (Blue-Yellow)", f"{roi_color['lab']['b_mean']:.2f}")
        else:
            st.info("No image uploaded. Please upload an image for analysis.")
    
    # Report Generation Tab
    with tabs[3]:
        st.subheader("Report Generation")
        
        # Check if analysis has been performed
        if ('prediction_result' in st.session_state and 
            st.session_state.prediction_result is not None and
            'texture_analysis' in st.session_state and
            'color_profile' in st.session_state):
            
            # Import the enhanced report generator
            from utils.report_generation import ReportGenerator
            
            # Create tabs for different report sections
            report_tabs = st.tabs(["Basic Info", "Performance Metrics", "Generate Report"])
            
            # Basic Info Tab
            with report_tabs[0]:
                # Patient information
                st.markdown("### Patient Information")
                patient_name = st.text_input("Patient Name", "")
                patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
                patient_gender = st.selectbox("Patient Gender", ["Male", "Female", "Other"])
                patient_notes = st.text_area("Additional Notes", "", help="Enter any relevant medical history or symptoms")
                
                # Display prediction result and confidence
                st.markdown("### Diagnosis Summary")
                st.info(f"**Predicted Condition:** {st.session_state.prediction_result} with {st.session_state.confidence_scores[st.session_state.prediction_result]*100:.2f}% confidence")
            
            # Performance Metrics Tab
            with report_tabs[1]:
                # Initialize report generator with current data
                report_gen = ReportGenerator(
                    prediction_result=st.session_state.prediction_result,
                    confidence_scores=st.session_state.confidence_scores
                )
                
                # Create visualization tabs
                viz_tabs = st.tabs([
                    "Accuracy & Loss", 
                    "Metrics Histogram", 
                    "Class Distribution", 
                    "Confusion Matrix", 
                    "Performance Metrics", 
                    "Metrics Table"
                ])
                
                # Accuracy and Loss Curves
                with viz_tabs[0]:
                    st.markdown("### Model Training Accuracy and Loss Curves")
                    accuracy_loss_img = report_gen.plot_accuracy_loss_curves()
                    st.image(
                        "data:image/png;base64," + accuracy_loss_img,
                        caption="Training and Validation Accuracy/Loss Curves",
                        use_container_width=True
                    )
                    st.markdown("""
                    These curves show how the model's accuracy improved and loss decreased during training.
                    - **Blue lines:** Training metrics
                    - **Red lines:** Validation metrics
                    
                    A good model shows increasing accuracy and decreasing loss that eventually stabilizes
                    without significant gaps between training and validation (which would indicate overfitting).
                    """)
                
                # Metrics Histogram
                with viz_tabs[1]:
                    st.markdown("### Distribution of Performance Metrics")
                    metrics_histogram = report_gen.plot_metrics_histogram()
                    st.image(
                        "data:image/png;base64," + metrics_histogram,
                        caption="Histogram of Precision, Recall, and F1 Scores",
                        use_container_width=True
                    )
                    st.markdown("""
                    This histogram shows the distribution of precision, recall, and F1 scores across all classes.
                    - Higher scores clustered toward the right indicate better overall performance
                    - Spread-out distributions may indicate inconsistent performance across different skin conditions
                    """)
                
                # Pie Chart
                with viz_tabs[2]:
                    st.markdown("### Class Distribution")
                    metrics_pie = report_gen.plot_metrics_pie_chart()
                    st.image(
                        "data:image/png;base64," + metrics_pie,
                        caption="Proportion of Correctly Classified Samples by Class",
                        use_container_width=True
                    )
                    st.markdown("""
                    This pie chart shows how the correctly classified samples are distributed across classes.
                    - Larger slices indicate more samples correctly identified for that class
                    - This helps identify which conditions the model is better at recognizing
                    """)
                
                # Confusion Matrix Heatmap
                with viz_tabs[3]:
                    st.markdown("### Confusion Matrix")
                    confusion_matrix = report_gen.plot_confusion_matrix_heatmap()
                    st.image(
                        "data:image/png;base64," + confusion_matrix,
                        caption="Confusion Matrix Heatmap",
                        use_container_width=True
                    )
                    st.markdown("""
                    The confusion matrix shows how samples from each true class were classified:
                    - Numbers on the diagonal are correctly classified samples
                    - Off-diagonal numbers show misclassifications
                    - Darker blue indicates higher numbers
                    
                    A good model has high numbers along the diagonal and low numbers elsewhere.
                    """)
                
                # Bar Chart Metrics
                with viz_tabs[4]:
                    st.markdown("### Performance Metrics by Class")
                    metrics_bar = report_gen.plot_metrics_bar_chart()
                    st.image(
                        "data:image/png;base64," + metrics_bar,
                        caption="Precision, Recall, and F1 Score for Each Class",
                        use_container_width=True
                    )
                    st.markdown("""
                    This chart compares precision, recall, and F1 scores across all skin conditions:
                    - **Precision:** Percentage of positive predictions that were correct
                    - **Recall:** Percentage of actual positives that were identified
                    - **F1 Score:** Harmonic mean of precision and recall
                    
                    Higher bars indicate better performance for that metric and class.
                    """)
                
                # Metrics Table
                with viz_tabs[5]:
                    st.markdown("### Detailed Performance Metrics Table")
                    metrics_df = report_gen.create_metrics_table()
                    st.dataframe(metrics_df, use_container_width=True)
                    st.markdown("""
                    This table provides detailed numerical values for all performance metrics:
                    - **Precision:** Indicates how many positively classified samples were actually positive
                    - **Recall:** Shows how many actual positives were correctly identified
                    - **F1 Score:** Balances precision and recall into a single metric
                    - **Support:** The number of samples in each class
                    """)
            
            # Generate Report Tab
            with report_tabs[2]:
                st.markdown("### Generate Comprehensive Report")
                st.markdown("""
                Generate a detailed PDF report with all analysis results, model performance metrics, 
                and visualizations. This report is suitable for medical professionals and contains:
                
                - Patient information
                - Diagnosis and confidence levels
                - Model performance metrics and visualizations
                - Detailed texture and color analysis
                - Region of interest analysis
                """)
                
                report_format = st.radio("Report Format", ["PDF", "HTML"], horizontal=True)
                
                # Generate report button
                if st.button("Generate Comprehensive Report", type="primary"):
                    if not patient_name:
                        st.warning("Please enter patient name in the Basic Info tab.")
                    else:
                        with st.spinner("Generating comprehensive report with all visualizations..."):
                            # Create patient info dictionary
                            patient_info = {
                                "Name": patient_name,
                                "Age": patient_age,
                                "Gender": patient_gender,
                                "Notes": patient_notes,
                                "Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                            }
                            
                            # Initialize enhanced report generator
                            report_gen = ReportGenerator(
                                prediction_result=st.session_state.prediction_result,
                                confidence_scores=st.session_state.confidence_scores
                            )
                            
                            if report_format == "PDF":
                                # Generate PDF report to memory
                                pdf_path = f"skin_analysis_report_{patient_name.replace(' ', '_')}.pdf"
                                
                                # Generate the enhanced PDF report with all visualizations
                                report_gen.generate_pdf_report(
                                    output_path=pdf_path,
                                    patient_info=patient_info,
                                    texture_analysis=st.session_state.texture_analysis,
                                    color_profile=st.session_state.color_profile
                                )
                                
                                # Read the PDF file and convert to base64
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                
                                # Offer download of PDF
                                st.success("Enhanced report with all visualizations generated successfully!")
                                
                                # Convert to base64 for download
                                b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                                
                                # Create download link
                                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_path}">Download Enhanced PDF Report</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                            else:  # HTML report
                                # Generate HTML report
                                html_report = report_gen.generate_html_report(
                                    patient_info=patient_info,
                                    texture_analysis=st.session_state.texture_analysis,
                                    color_profile=st.session_state.color_profile
                                )
                                
                                # Offer download of HTML
                                st.success("Enhanced HTML report with all visualizations generated successfully!")
                                
                                # Convert to base64 for download
                                b64_html = base64.b64encode(html_report.encode('utf-8')).decode('utf-8')
                                
                                # Create download link
                                href = f'<a href="data:text/html;base64,{b64_html}" download="skin_analysis_report_{patient_name.replace(" ", "_")}.html">Download Enhanced HTML Report</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                                # Preview HTML report
                                st.markdown("### HTML Report Preview")
                                st.components.v1.html(html_report, height=600, scrolling=True)
                
                # Also offer basic report option
                st.markdown("### Generate Basic Report (Legacy)")
                if st.button("Generate Basic Report"):
                    if not patient_name:
                        st.warning("Please enter patient name in the Basic Info tab.")
                    else:
                        with st.spinner("Generating basic report..."):
                            # Get all required data
                            prediction_result = st.session_state.prediction_result
                            confidence_scores = st.session_state.confidence_scores
                            uploaded_image = st.session_state.uploaded_image
                            texture_analysis = st.session_state.texture_analysis
                            color_profile = st.session_state.color_profile
                            confidence_plot = st.session_state.get('confidence_plot')
                            grad_cam = st.session_state.get('grad_cam')
                            
                            # Import the legacy report generator
                            from utils.report_generator import create_report
                            
                            # Generate PDF report
                            pdf_bytes = create_report(
                                patient_name,
                                prediction_result,
                                confidence_scores,
                                uploaded_image,
                                texture_analysis,
                                color_profile,
                                confidence_plot,
                                grad_cam
                            )
                            
                            # Offer download of PDF
                            st.success("Basic report generated successfully!")
                            
                            # Convert to base64 for download
                            b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                            
                            # Create download link
                            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="skin_analysis_report_basic_{patient_name.replace(" ", "_")}.pdf">Download Basic PDF Report</a>'
                            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No analysis results available. Please upload and analyze an image first.")
