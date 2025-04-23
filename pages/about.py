import streamlit as st

def show_about():
    """
    Display the about page with information about the application.
    """
    st.title("About SkinDiagnose AI")
    st.markdown("---")
    
    # App description
    st.markdown("""
    ### Overview
    
    SkinDiagnose AI is an advanced skin disease analysis tool that leverages deep learning and computer vision technologies to help identify and analyze various skin conditions. The application is designed to assist healthcare professionals and patients in understanding skin diseases through automated analysis.
    
    ### Features
    
    - **Disease Classification**: Identifies 5 common skin conditions:
      - Acne
      - Hyperpigmentation
      - Nail Psoriasis
      - Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis (SJS-TEN)
      - Vitiligo
    
    - **Advanced Analysis**:
      - Texture analysis of skin lesions
      - Color profile examination
      - Region of Interest (ROI) selection and analysis
    
    - **Reporting**:
      - Comprehensive PDF report generation
      - Detailed metrics and visualizations
    
    - **AI Chatbot**:
      - Interactive information about skin conditions
      - Powered by a local language model
    """)
    
    # Create columns for technology stack and limitations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Technology Stack
        
        - **Frontend**: Streamlit
        - **Image Processing**: OpenCV, scikit-image
        - **Machine Learning**: PyTorch, TorchVision
        - **Visualization**: Matplotlib, Plotly
        - **Report Generation**: ReportLab
        - **AI Assistant**: Custom local language model
        """)
    
    with col2:
        st.markdown("""
        ### Limitations
        
        - This tool is for educational and informational purposes only
        - Not a substitute for professional medical diagnosis
        - Accuracy depends on image quality and clarity
        - Limited to analyzing 5 specific skin conditions
        - Results should be interpreted by healthcare professionals
        """)
    
    st.markdown("---")
    
    # Medical disclaimer
    st.subheader("Medical Disclaimer")
    st.info("""
    The content provided by SkinDiagnose AI is for informational purposes only and does not constitute medical advice, diagnosis, or treatment recommendations. Always consult with a qualified healthcare provider for proper diagnosis and treatment of medical conditions.
    
    SkinDiagnose AI is intended to be used as a tool to assist healthcare professionals and is not designed to replace professional medical judgment. The creators of SkinDiagnose AI are not responsible for any health problems that may result from using this application.
    """)
    
    # References and resources
    st.markdown("""
    ### References and Resources
    
    - American Academy of Dermatology: [https://www.aad.org/](https://www.aad.org/)
    - National Psoriasis Foundation: [https://www.psoriasis.org/](https://www.psoriasis.org/)
    - Vitiligo Support International: [https://www.vitiligosupport.org/](https://www.vitiligosupport.org/)
    - Genetic and Rare Diseases Information Center (for SJS-TEN): [https://rarediseases.info.nih.gov/](https://rarediseases.info.nih.gov/)
    """)
    
    # Contact information
    st.markdown("""
    ### Contact Information
    
    For questions, feedback, or support regarding SkinDiagnose AI, please contact:
    
    **Email**: support@skindiagnose.ai  
    **Website**: www.skindiagnose.ai
    
    *Note: This is a demonstration application. The contact information is fictional and for illustrative purposes only.*
    """)
