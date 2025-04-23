import json
import os
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user

def get_openai_client():
    """
    Get the OpenAI client using the API key from environment variables.
    
    Returns:
        OpenAI client or None if API key is not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    
    return OpenAI(api_key=api_key)

def get_medication_recommendations(condition: str, severity: str = "moderate", 
                                  patient_info: dict = None):
    """
    Get personalized medication recommendations for a skin condition using OpenAI.
    
    Args:
        condition: The skin condition name
        severity: Severity level (mild, moderate, severe)
        patient_info: Optional dictionary with patient information like age, allergies, etc.
        
    Returns:
        Dictionary with medication recommendations and precautions
    """
    client = get_openai_client()
    if not client:
        return {"error": "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."}
    
    # Default patient info if not provided
    if patient_info is None:
        patient_info = {
            "age_group": "adult",
            "allergies": [],
            "medical_history": []
        }
    
    try:
        # Create the prompt
        prompt = f"""
        As a dermatology expert, provide medication recommendations for {condition} with {severity} severity.
        
        Patient information:
        - Age group: {patient_info.get('age_group', 'adult')}
        - Allergies: {', '.join(patient_info.get('allergies', ['None reported']))}
        - Medical history: {', '.join(patient_info.get('medical_history', ['None reported']))}
        
        Format your response as a JSON object with the following structure:
        {{
            "over_the_counter": [
                {{
                    "name": "Medication name",
                    "active_ingredient": "Active ingredient",
                    "dosage": "Recommended dosage",
                    "frequency": "How often to use",
                    "precautions": "Important precautions"
                }}
            ],
            "prescription": [
                {{
                    "name": "Medication name",
                    "active_ingredient": "Active ingredient",
                    "typical_dosage": "Typical prescription dosage",
                    "precautions": "Important precautions"
                }}
            ],
            "lifestyle_recommendations": [
                "Recommendation 1",
                "Recommendation 2"
            ],
            "general_advice": "General advice for this condition"
        }}
        
        Provide at least 2-3 options for each medication category when applicable, and focus on evidence-based treatments.
        """
        
        # Get the response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        return {"error": f"Error getting medication recommendations: {str(e)}"}

def analyze_image_with_openai(base64_image):
    """
    Analyze a skin image using OpenAI's vision capabilities.
    
    Args:
        base64_image: Base64 encoded image
        
    Returns:
        Dictionary with analysis results
    """
    client = get_openai_client()
    if not client:
        return {"error": "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."}
    
    try:
        # Create the prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this skin image as a dermatologist would. Identify any visible skin conditions, "
                                   "describe the appearance, possible diagnoses, and recommended next steps. Format your "
                                   "response as JSON with keys: description, potential_conditions, severity_assessment, "
                                   "and recommendations."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        return {"error": f"Error analyzing image: {str(e)}"}