import streamlit as st
import time
import io
from PIL import Image
from utils.enhanced_llm import EnhancedLLMManager
from utils.model_utils import load_model, predict_disease
from utils.image_processing import preprocess_image, analyze_texture, analyze_color_profile
from utils.model_utils import generate_grad_cam, plot_confidence_scores

def get_condition_info(condition: str) -> str:
    """Get detailed information about a skin condition."""
    condition_info = {
        "Acne": """
        Acne is a skin condition that occurs when hair follicles plug with oil and dead skin cells.
        It causes whiteheads, blackheads, or pimples and typically appears on the face, forehead, 
        chest, upper back, and shoulders. Acne is most common among teenagers but affects people 
        of all ages. It's characterized by inflammation, redness, and sometimes painful lesions.
        """,
        
        "Hyperpigmentation": """
        Hyperpigmentation is a common condition where patches of skin become darker than the 
        surrounding area due to excess melanin production. It can be caused by sun damage, 
        inflammation, hormonal changes, or certain medications. Common types include age spots, 
        melasma, and post-inflammatory hyperpigmentation. It affects people of all skin types 
        but is more noticeable in those with darker skin tones.
        """,
        
        "Nail Psoriasis": """
        Nail psoriasis is a manifestation of psoriasis affecting the fingernails and toenails.
        It causes pitting, discoloration, abnormal nail growth, separation of the nail from the 
        nail bed, and crumbling of the nail. About 50% of people with psoriasis have nail 
        involvement, and this figure rises to 80% for those with psoriatic arthritis. It can 
        be painful and affect daily activities.
        """,
        
        "SJS-TEN": """
        Stevens-Johnson syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) are severe, 
        potentially life-threatening skin reactions. They're usually caused by medication 
        and characterized by extensive skin detachment and mucosal involvement. Symptoms 
        include fever, skin pain, and a widespread rash that blisters and causes the top 
        layer of the skin to die and shed. Immediate medical attention is critical.
        """,
        
        "Vitiligo": """
        Vitiligo is a chronic condition where skin loses its pigmentation, resulting in 
        white patches. It occurs when melanocytes, the cells responsible for skin color, 
        are destroyed. The exact cause is unknown, but it's thought to be an autoimmune 
        disorder. Vitiligo can affect any part of the body and may spread over time. It's 
        not physically harmful but can cause psychological distress.
        """
    }
    
    return condition_info.get(condition, "Information not available for this condition.")

def get_treatment_recommendations(condition: str) -> str:
    """Get treatment recommendations for a skin condition."""
    treatment_recommendations = {
        "Acne": """
        1. **Daily Skincare Routine**:
           - Wash affected areas twice daily with a gentle cleanser
           - Use non-comedogenic moisturizers and sunscreen
        
        2. **Over-the-counter treatments**:
           - Products containing benzoyl peroxide to kill bacteria
           - Salicylic acid to help unclog pores
           - Adapalene (Differin) for mild to moderate acne
        
        3. **Prescription options** (consult a dermatologist):
           - Topical retinoids
           - Antibiotics (topical or oral)
           - Hormonal therapies for women
           - Isotretinoin for severe acne
           
        4. **Lifestyle factors**:
           - Avoid touching or picking at acne
           - Maintain a healthy diet and stay hydrated
           - Manage stress through regular exercise and sleep
        """,
        
        "Hyperpigmentation": """
        1. **Prevention**:
           - Apply broad-spectrum sunscreen (SPF 30+) daily
           - Wear protective clothing and seek shade
           - Avoid picking at skin inflammation or injuries
        
        2. **Topical treatments**:
           - Products with ingredients like vitamin C, niacinamide, or alpha arbutin
           - Exfoliants with AHAs or BHAs to promote cell turnover
           - Retinoids to accelerate cell regeneration
        
        3. **Professional treatments** (consult a dermatologist):
           - Chemical peels
           - Microdermabrasion
           - Laser therapy
           - Intense pulsed light (IPL)
        
        4. **For hormonal hyperpigmentation** (melasma):
           - Hydroquinone (under medical supervision)
           - Triple combination creams
           - Oral tranexamic acid in some cases
        """,
        
        "Nail Psoriasis": """
        1. **Topical treatments**:
           - Corticosteroids to reduce inflammation
           - Vitamin D analogues to slow skin cell growth
           - Tacrolimus or pimecrolimus for sensitive areas
        
        2. **Systemic treatments** (for severe cases):
           - Oral medications like methotrexate or cyclosporine
           - Biologic medications that target specific immune pathways
        
        3. **Nail care tips**:
           - Keep nails trimmed short
           - Avoid trauma to the nails
           - Use moisturizers containing urea to soften nails
           - Consider using clear nail polish to protect nails
        
        4. **Lifestyle modifications**:
           - Quit smoking (it can worsen psoriasis)
           - Maintain a healthy weight
           - Manage stress through relaxation techniques
        """,
        
        "SJS-TEN": """
        **URGENT: Seek emergency medical care immediately!**
        
        1. **Immediate actions**:
           - Discontinue all non-essential medications
           - Go to the emergency room or call emergency services
           - SJS-TEN is a medical emergency requiring hospitalization
        
        2. **Hospital treatment**:
           - Supportive care in a burn unit or intensive care
           - Fluid and electrolyte management
           - Wound care and pain management
           - Monitoring for infections and complications
        
        3. **Post-recovery**:
           - Document the medication that caused the reaction
           - Wear a medical alert bracelet
           - Follow up with specialists for potential long-term complications
           - Genetic counseling may be recommended in some cases
        
        4. **Prevention of recurrence**:
           - Avoid the triggering medication and related drugs
           - Inform all healthcare providers about your history
           - Consult with a specialist before taking new medications
        """,
        
        "Vitiligo": """
        1. **Medical treatments**:
           - Topical corticosteroids to reduce inflammation
           - Calcineurin inhibitors like tacrolimus
           - Phototherapy (UVB therapy or PUVA)
           - Excimer laser for small areas
        
        2. **For extensive vitiligo**:
           - Oral or injectable corticosteroids (short courses)
           - JAK inhibitors (newer treatments)
           - Surgical options like skin grafting for stable vitiligo
        
        3. **Cosmetic options**:
           - Sunscreen to protect affected areas
           - Self-tanners to even skin tone
           - Makeup and body makeup products
           - Medical tattooing for smaller areas
        
        4. **Supportive measures**:
           - Sun protection to prevent contrast between affected and unaffected skin
           - Vitamin D supplementation if recommended
           - Psychological support or counseling if needed
           - Support groups for emotional well-being
        """
    }
    
    return treatment_recommendations.get(condition, "Treatment information not available for this condition.")

def show_chatbot():
    """
    Display the AI chatbot page.
    """
    st.title("Skin Disease AI Chatbot")
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ### Ask questions about ANY skin condition
    
    This AI assistant can provide real-time information about various skin conditions including:
    - Acne, Rosacea, and other inflammatory conditions
    - Hyperpigmentation, Vitiligo, and other pigmentation disorders
    - Psoriasis, Eczema, and other chronic skin conditions
    - Fungal infections, Warts, and other contagious skin issues
    - Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis (SJS-TEN) and other serious reactions
    - And many more skin conditions!
    
    Ask about symptoms, treatments, causes, prevention, or compare different conditions. The chatbot provides real-time, knowledge-based responses to all your skin health questions using a local LLM that doesn't require internet access.
    
    For a more personalized experience, try uploading a skin image in the sidebar to get a real-time diagnosis with treatment recommendations.
    """)
    
    # Initialize LLM manager if not already in session state
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = EnhancedLLMManager()
        st.session_state.llm_manager.load_model()
    
    # Initialize chat history if not already in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about skin conditions..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant message with typing effect
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for token in st.session_state.llm_manager.generate_stream(prompt):
                full_response += token
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    
    # Sidebar for chat options
    with st.sidebar:
        st.markdown("### Chatbot Options")
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            # Also clear any image analysis results
            if 'chat_uploaded_image' in st.session_state:
                del st.session_state.chat_uploaded_image
            if 'chat_prediction_result' in st.session_state:
                del st.session_state.chat_prediction_result
            if 'chat_confidence_scores' in st.session_state:
                del st.session_state.chat_confidence_scores
            st.rerun()
        
        # Image upload for diagnosis
        st.markdown("### Upload Image for Diagnosis")
        st.markdown("Upload a skin image to get a real-time diagnosis and personalized advice.")
        
        uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Display image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Button to analyze the image
            if st.button("Analyze Image"):
                # Show processing message
                with st.spinner("Analyzing image..."):
                    try:
                        # Load model
                        model = load_model()
                        
                        # Perform prediction
                        predicted_class, confidence_scores = predict_disease(image, model)
                        
                        # Process image features
                        texture_analysis = analyze_texture(image)
                        color_profile = analyze_color_profile(image)
                        
                        # Generate Grad-CAM and confidence plot
                        grad_cam_base64 = generate_grad_cam(image, model)
                        confidence_plot = plot_confidence_scores(confidence_scores)
                        
                        # Store results in session state
                        st.session_state.chat_uploaded_image = image
                        st.session_state.chat_prediction_result = predicted_class
                        st.session_state.chat_confidence_scores = confidence_scores
                        st.session_state.chat_texture_analysis = texture_analysis
                        st.session_state.chat_color_profile = color_profile
                        st.session_state.chat_grad_cam = grad_cam_base64
                        st.session_state.chat_confidence_plot = confidence_plot
                        
                        # Generate a message about the prediction for the chat
                        result_message = f"Based on my analysis, this appears to be **{predicted_class}** with a confidence of {confidence_scores[predicted_class]*100:.2f}%."
                        
                        # Generate real-time information using the local LLM
                        condition_prompt = f"Give detailed information about {predicted_class} skin condition, including symptoms, causes, and basic treatments."
                        condition_details = st.session_state.llm_manager.generate_response(condition_prompt)
                        
                        # Add result as user and assistant messages to the chat
                        st.session_state.chat_history.append({
                            "role": "user", 
                            "content": "Can you analyze this skin image and suggest treatments?"
                        })
                        
                        # Build the response based on available data
                        full_response = f"{result_message}\n\n"
                        
                        # Add AI analysis from local LLM
                        full_response += f"### About {predicted_class}:\n{condition_details}\n\n"
                        
                        # Add treatment recommendations
                        full_response += f"### Recommended next steps:\n{get_treatment_recommendations(predicted_class)}\n\n"
                        
                        # Add texture and color analysis information
                        if texture_analysis and color_profile:
                            full_response += "### Image Analysis Details:\n\n"
                            
                            # Texture analysis
                            full_response += "#### Texture Analysis:\n"
                            full_response += f"- Contrast: {texture_analysis.get('contrast', 'N/A'):.4f}\n"
                            full_response += f"- Homogeneity: {texture_analysis.get('homogeneity', 'N/A'):.4f}\n"
                            full_response += f"- Energy: {texture_analysis.get('energy', 'N/A'):.4f}\n"
                            full_response += f"- Correlation: {texture_analysis.get('correlation', 'N/A'):.4f}\n\n"
                            
                            # Color profile
                            full_response += "#### Color Profile:\n"
                            full_response += f"- Average Redness: {color_profile.get('avg_red', 0):.2f}%\n"
                            full_response += f"- Average Saturation: {color_profile.get('avg_saturation', 0):.2f}%\n"
                            full_response += f"- Color Variance: {color_profile.get('color_variance', 0):.2f}\n\n"
                            
                        # Add disclaimer
                        full_response += "*Disclaimer: This analysis is for informational purposes only. Always consult with a healthcare professional for accurate diagnosis and treatment.*"
                        
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
        
        # Example questions
        st.markdown("### Example Questions")
        example_questions = [
            "What are the symptoms of eczema?",
            "How is rosacea treated?",
            "What causes fungal infections on skin?",
            "Compare psoriasis and eczema",
            "How to prevent warts?",
            "What are the early signs of skin cancer?",
            "Is ringworm contagious and how is it treated?"
        ]
        
        for question in example_questions:
            if st.button(question):
                # Set the question as the current input and trigger chat
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(question)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for token in st.session_state.llm_manager.generate_stream(question):
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)
                    
                    message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                st.rerun()
        
        # Disclaimer
        st.markdown("---")
        st.caption("""
        **Disclaimer:** This chatbot provides general information about skin conditions and is not a substitute for professional medical advice. Always consult with a healthcare provider for medical concerns.
        """)
