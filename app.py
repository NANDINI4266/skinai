
import streamlit as st
import os
import sys
import asyncio

# Fix for asyncio event loop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="SkinDiagnose AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import pages
from pages.home import show_home
from pages.analyzer import show_analyzer
from pages.chatbot import show_chatbot
from pages.about import show_about
from pages.model_evaluation import show_model_evaluation

# Initialize session state variables if they don't exist
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence_scores' not in st.session_state:
    st.session_state.confidence_scores = None
if 'roi_selection' not in st.session_state:
    st.session_state.roi_selection = None
if 'texture_analysis' not in st.session_state:
    st.session_state.texture_analysis = None
if 'color_profile' not in st.session_state:
    st.session_state.color_profile = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar navigation
st.sidebar.title("SkinDiagnose AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to", 
    ["Home", "Disease Analyzer", "Model Evaluation", "AI Chatbot", "About"],
    index=0
)

# Show information about the enhanced local LLM
st.sidebar.markdown("---")
st.sidebar.markdown("### Enhanced Local LLM Features")
st.sidebar.info("""
This application uses a real-time local knowledge-based LLM to provide instant responses 
about skin conditions. No external APIs or internet connection required.

The enhanced LLM features include:
- Information about a wide range of skin conditions
- Comparison between different conditions
- Detailed symptoms, treatments, and prevention advice
- Real-time streaming response generation
""")

# Display selected page
if page == "Home":
    show_home()
elif page == "Disease Analyzer":
    show_analyzer()
elif page == "Model Evaluation":
    show_model_evaluation()
elif page == "AI Chatbot":
    show_chatbot()
elif page == "About":
    show_about()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2023 SkinDiagnose AI")
st.sidebar.caption("This application is for educational purposes only and should not replace professional medical advice.")
