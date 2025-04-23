import streamlit as st

def show_home():
    """
    Display the home page of the application.
    """
    # Main title and description
    st.title("SkinDiagnose AI")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to SkinDiagnose AI - Advanced Skin Disease Analysis
    
    SkinDiagnose AI leverages cutting-edge machine learning technology to help identify and analyze various skin conditions.
    
    **Our system can detect:**
    - Acne
    - Hyperpigmentation
    - Nail Psoriasis
    - Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis (SJS-TEN)
    - Vitiligo
    
    **Advanced features include:**
    - Real-time disease classification
    - Detailed texture analysis
    - Region of Interest (ROI) selection
    - Color profile analysis
    - Model accuracy metrics and performance visualization
    - Comprehensive report generation
    - AI-powered chatbot for skin disease information
    """)
    
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    # First column - Getting Started
    with col1:
        st.subheader("Getting Started")
        st.markdown("""
        1. Navigate to the **Disease Analyzer** page
        2. Upload an image of the skin condition
        3. View the prediction results and analysis
        4. Generate a detailed report if needed
        """)
    
    # Second column - Important Notice
    with col2:
        st.subheader("Important Notice")
        st.markdown("""
        This application is for educational and informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
        
        Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
        """)
    
    st.markdown("---")
    
    # How It Works section
    st.subheader("How It Works")
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1. Upload Image")
        st.markdown("Upload a clear image of the skin condition. Ensure good lighting and focus for best results.")
    
    with col2:
        st.markdown("#### 2. AI Analysis")
        st.markdown("Our AI system analyzes the image using advanced computer vision and deep learning techniques.")
    
    with col3:
        st.markdown("#### 3. Results & Report")
        st.markdown("View detailed analysis results and generate a comprehensive report if needed.")
    
    st.markdown("---")
    
    # Call to action
    st.subheader("Ready to Start?")
    
    # Create two columns for action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Go to Disease Analyzer", type="primary", use_container_width=True):
            # This will set the page to the analyzer page
            st.session_state.page = "Disease Analyzer"
            st.rerun()
    
    with col2:
        if st.button("View Model Metrics", use_container_width=True):
            # This will set the page to the model evaluation page
            st.session_state.page = "Model Evaluation"
            st.rerun()
