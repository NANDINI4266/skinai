import os
import numpy as np
import json
import time
import threading
import queue
import re
import math
import random
from typing import List, Dict, Generator, Tuple

class EnhancedSkinLLM:
    """
    An enhanced real-time, knowledge-based LLM for skin disease information.
    This implementation provides more comprehensive coverage of skin conditions
    with improved language generation capabilities.
    """
    
    def __init__(self, model_path=None, context_size=512, threads=4):
        """
        Initialize the EnhancedSkinLLM model.
        
        Args:
            model_path: Path to knowledge base files (if any)
            context_size: Memory context size
            threads: Number of threads for parallel processing
        """
        self.context_size = context_size
        self.threads = threads
        self.model_loaded = False
        
        # Load expanded knowledge graph for skin conditions
        self._initialize_knowledge_graph()
        
        # Initialize token bank and language patterns
        self._initialize_token_bank()
        self._initialize_language_patterns()
        
        # Set model to loaded state
        self.model_loaded = True
        
    def _initialize_knowledge_graph(self):
        """Initialize the comprehensive knowledge graph for skin conditions"""
        
        # Expanded skin conditions knowledge base
        self.knowledge_graph = {
            "acne": {
                "description": "A skin condition that occurs when hair follicles plug with oil and dead skin cells.",
                "symptoms": ["Whiteheads", "Blackheads", "Pimples", "Cysts", "Inflammation", "Redness", "Oily skin", "Scarring"],
                "treatments": ["Benzoyl peroxide", "Salicylic acid", "Retinoids", "Antibiotics", "Isotretinoin", "Hormonal treatments", "Light therapy"],
                "prevention": ["Regular cleansing", "Avoiding heavy makeup", "Not touching face", "Balanced diet", "Stress management", "Using non-comedogenic products"],
                "causes": ["Excess oil production", "Bacteria", "Hormonal changes", "Genetics", "Certain medications"],
                "complications": ["Scarring", "Hyperpigmentation", "Psychological distress"],
                "affected_areas": ["Face", "Chest", "Back", "Shoulders"],
                "statistics": "Affects up to 50 million Americans annually, making it the most common skin condition in the United States.",
                "related_conditions": ["Rosacea", "Folliculitis", "Hidradenitis suppurativa"]
            },
            "hyperpigmentation": {
                "description": "A condition where patches of skin become darker in color than the surrounding skin due to excess melanin.",
                "symptoms": ["Dark patches", "Uneven skin tone", "Age spots", "Melasma", "Post-inflammatory marks", "Sun spots"],
                "treatments": ["Hydroquinone", "Vitamin C", "Chemical peels", "Laser therapy", "Retinoids", "Azelaic acid", "Kojic acid", "Microdermabrasion"],
                "prevention": ["Sun protection", "Gentle skin care", "Avoiding trauma to skin", "Managing hormonal changes", "Using broad-spectrum sunscreen"],
                "causes": ["Sun exposure", "Inflammation", "Hormonal changes", "Medications", "Wound healing", "Genetics"],
                "complications": ["Psychological distress", "Cosmetic concerns", "Self-esteem issues"],
                "affected_areas": ["Face", "Hands", "Neck", "Any sun-exposed area"],
                "statistics": "Melasma affects up to 6 million people in the United States, with 90% being women.",
                "related_conditions": ["Vitiligo", "Albinism", "Freckles", "Post-inflammatory erythema"]
            },
            "nail_psoriasis": {
                "description": "A manifestation of psoriasis affecting the fingernails and toenails, causing various nail changes.",
                "symptoms": ["Pitting", "Discoloration", "Crumbling", "Separation from nail bed", "Abnormal growth", "Thickening", "Ridges"],
                "treatments": ["Topical corticosteroids", "Vitamin D analogs", "Light therapy", "Biologics", "Systemic medications", "Intralesional injections"],
                "prevention": ["Avoiding trauma to nails", "Proper nail care", "Keeping nails short", "Managing stress", "Treating underlying psoriasis"],
                "causes": ["Autoimmune disorder", "Genetic factors", "Environmental triggers", "Stress", "Infections"],
                "complications": ["Pain", "Difficulty using hands/walking", "Infections", "Psychological impact"],
                "affected_areas": ["Fingernails", "Toenails"],
                "statistics": "Affects about 50% of people with psoriasis and 80-90% of those with psoriatic arthritis.",
                "related_conditions": ["Psoriasis", "Psoriatic arthritis", "Fungal nail infections"]
            },
            "sjs_ten": {
                "description": "Stevens-Johnson syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) are severe, potentially life-threatening skin reactions usually caused by medications.",
                "symptoms": ["Fever", "Skin pain", "Blistering", "Skin peeling", "Mucosal involvement", "Red or purple rash", "Eye inflammation"],
                "treatments": ["Immediate medication discontinuation", "Supportive care", "Fluid management", "Wound care", "Eye care", "Immune-modulating treatments"],
                "prevention": ["Avoiding trigger medications", "Genetic testing if family history", "Medical alert bracelet", "Informing all healthcare providers"],
                "causes": ["Medications (antibiotics, anticonvulsants, NSAIDs)", "Infections", "Genetic predisposition"],
                "complications": ["Sepsis", "Shock", "Death", "Long-term skin problems", "Eye damage", "Lung damage"],
                "affected_areas": ["Entire body", "Mucous membranes (eyes, mouth, genitals)"],
                "statistics": "SJS/TEN affects 1-2 persons per million annually. Mortality rate is 10% for SJS and 30-50% for TEN.",
                "related_conditions": ["Erythema multiforme", "Drug reactions", "Staphylococcal scalded skin syndrome"]
            },
            "vitiligo": {
                "description": "A condition where skin loses its pigment cells (melanocytes), resulting in white patches on different areas of the body.",
                "symptoms": ["White patches on skin", "Premature whitening of hair", "Loss of color in the mouth", "Loss of eye color"],
                "treatments": ["Topical corticosteroids", "Calcineurin inhibitors", "Light therapy", "Surgery", "Depigmentation", "Cosmetic camouflage"],
                "prevention": ["No known prevention", "Sun protection", "Minimizing physical trauma to skin"],
                "causes": ["Autoimmune disorder", "Genetic factors", "Environmental triggers", "Stress", "Neural factors"],
                "complications": ["Sunburn", "Eye problems", "Hearing problems", "Psychological distress", "Social stigma"],
                "affected_areas": ["Face", "Hands", "Armpits", "Groin", "Around body openings", "Can be anywhere"],
                "statistics": "Affects 1-2% of the world's population, with equal prevalence across genders and racial groups.",
                "related_conditions": ["Autoimmune thyroid disease", "Addison's disease", "Pernicious anemia", "Alopecia areata"]
            },
            
            # Additional skin conditions for enhanced coverage
            "eczema": {
                "description": "A group of conditions that cause the skin to become itchy, inflamed, or have a rash-like appearance.",
                "symptoms": ["Itching", "Dry skin", "Redness", "Rash", "Scaly patches", "Thickened skin", "Vesicles (small blisters)"],
                "treatments": ["Moisturizers", "Topical corticosteroids", "Topical calcineurin inhibitors", "Antihistamines", "Phototherapy", "Systemic immunosuppressants", "Biologics"],
                "prevention": ["Regular moisturizing", "Avoiding triggers", "Gentle skin care", "Humidifier use", "Cotton clothing", "Avoiding harsh detergents"],
                "causes": ["Genetic factors", "Immune system dysfunction", "Environmental factors", "Irritants", "Allergens", "Stress"],
                "complications": ["Skin infections", "Sleep problems", "Asthma and hay fever", "Quality of life impact"],
                "affected_areas": ["Face", "Hands", "Feet", "Inside elbows", "Behind knees", "Scalp"],
                "statistics": "Affects up to 20% of children and 10% of adults worldwide.",
                "related_conditions": ["Asthma", "Hay fever", "Food allergies", "Contact dermatitis"]
            },
            "rosacea": {
                "description": "A chronic inflammatory skin condition that primarily affects the face, causing redness, visible blood vessels, and sometimes small red bumps.",
                "symptoms": ["Facial redness", "Visible blood vessels", "Swollen red bumps", "Eye problems", "Enlarged nose", "Burning or stinging", "Sensitive skin"],
                "treatments": ["Topical medications (metronidazole, azelaic acid)", "Oral antibiotics", "Isotretinoin", "Laser treatment", "Light therapy"],
                "prevention": ["Avoiding triggers", "Sun protection", "Gentle skin care", "Avoiding hot drinks", "Stress management"],
                "causes": ["Blood vessel abnormalities", "Demodex mites", "Genetics", "Immune system factors", "Environmental factors"],
                "complications": ["Permanent skin damage", "Eye damage", "Psychological distress", "Rhinophyma (enlarged nose)"],
                "affected_areas": ["Cheeks", "Nose", "Forehead", "Chin", "Sometimes eyes"],
                "statistics": "Affects approximately 16 million Americans and over 415 million people worldwide.",
                "related_conditions": ["Acne", "Seborrheic dermatitis", "Lupus", "Steroid rosacea"]
            },
            "psoriasis": {
                "description": "A chronic autoimmune condition that causes rapid buildup of skin cells, resulting in scaling and inflammation.",
                "symptoms": ["Red patches with silvery scales", "Dry, cracked skin", "Itching", "Burning", "Soreness", "Thickened nails", "Joint pain"],
                "treatments": ["Topical corticosteroids", "Vitamin D analogs", "Retinoids", "Light therapy", "Methotrexate", "Cyclosporine", "Biologics"],
                "prevention": ["Stress management", "Avoiding skin injuries", "Avoiding medications that trigger flares", "Not smoking", "Moderate sun exposure"],
                "causes": ["Immune system dysfunction", "Genetics", "Environmental triggers", "Stress", "Infections"],
                "complications": ["Psoriatic arthritis", "Eye conditions", "Obesity", "Type 2 diabetes", "Cardiovascular disease"],
                "affected_areas": ["Scalp", "Elbows", "Knees", "Lower back", "Nails", "Can affect any area"],
                "statistics": "Affects about 8 million Americans and 125 million people worldwide.",
                "related_conditions": ["Psoriatic arthritis", "Inflammatory bowel disease", "Cardiovascular disease", "Depression"]
            },
            "fungal_infections": {
                "description": "Infections caused by fungi that can affect the skin, nails, and mucous membranes.",
                "symptoms": ["Itching", "Red rash", "Scaling", "Cracking", "Burning", "Nail discoloration", "Thickened nails"],
                "treatments": ["Antifungal creams", "Oral antifungal medications", "Medicated shampoos", "Nail removal (severe cases)"],
                "prevention": ["Keeping skin clean and dry", "Avoiding sharing personal items", "Wearing breathable materials", "Using antifungal powders"],
                "causes": ["Dermatophytes", "Yeasts", "Molds", "Warm, moist environments", "Compromised immunity"],
                "complications": ["Bacterial infections", "Widespread infection", "Cellulitis", "Hair loss", "Nail damage"],
                "affected_areas": ["Feet", "Groin", "Scalp", "Body folds", "Nails", "Mouth"],
                "statistics": "Fungal skin infections affect about 25% of the world's population at any given time.",
                "related_conditions": ["Diabetes", "Immunosuppression", "Obesity", "Hyperhidrosis"]
            },
            "warts": {
                "description": "Small, grainy skin growths caused by human papillomavirus (HPV) infection.",
                "symptoms": ["Rough, raised bumps", "Black pinpoints (clotted blood vessels)", "Discomfort or pain", "Itching"],
                "treatments": ["Salicylic acid", "Cryotherapy", "Duct tape", "Electrosurgery", "Laser treatment", "Immunotherapy"],
                "prevention": ["Avoiding direct contact with warts", "Not picking at warts", "Keeping hands clean", "Wearing footwear in public areas"],
                "causes": ["HPV infection", "Direct contact", "Breaks in skin barrier", "Compromised immunity"],
                "complications": ["Spread to other body parts", "Spread to other people", "Pain when walking (plantar warts)"],
                "affected_areas": ["Hands", "Feet", "Face", "Genitals", "Can occur anywhere"],
                "statistics": "About 10% of people have warts at any given time, with children and young adults most commonly affected.",
                "related_conditions": ["HPV-related cancers", "Immunodeficiency disorders"]
            }
        }
        
        # Define common skin symptoms and their associated conditions for more flexible matching
        self.symptom_to_condition = {
            "pimples": ["acne", "rosacea", "folliculitis"],
            "itching": ["eczema", "psoriasis", "fungal_infections", "hives", "contact dermatitis"],
            "redness": ["rosacea", "eczema", "psoriasis", "contact dermatitis", "sunburn"],
            "rash": ["eczema", "psoriasis", "contact dermatitis", "drug reaction", "viral exanthem"],
            "blisters": ["herpes", "chickenpox", "contact dermatitis", "bullous pemphigoid", "sjs_ten"],
            "patches": ["vitiligo", "psoriasis", "eczema", "ringworm", "pityriasis rosea"],
            "scaling": ["psoriasis", "seborrheic dermatitis", "fungal_infections", "ichthyosis", "eczema"],
            "bumps": ["acne", "warts", "molluscum contagiosum", "folliculitis", "keratosis pilaris"],
            "discoloration": ["hyperpigmentation", "vitiligo", "melasma", "bruising", "post-inflammatory changes"]
        }
        
    def _initialize_token_bank(self):
        """Initialize token bank for language generation from knowledge graph"""
        # Sentence starters for different response types
        self.response_templates = {
            "description": [
                "{condition} is {description}",
                "{condition} refers to {description}",
                "{condition} can be described as {description}",
                "Medical professionals define {condition} as {description}",
                "In dermatology, {condition} is classified as {description}"
            ],
            "symptoms": [
                "Common symptoms of {condition} include {symptoms}.",
                "People with {condition} often experience {symptoms}.",
                "The main signs to look for with {condition} are {symptoms}.",
                "If you have {condition}, you might notice {symptoms}.",
                "{condition} typically presents with {symptoms}."
            ],
            "treatments": [
                "Treatment options for {condition} include {treatments}.",
                "Dermatologists often recommend {treatments} for managing {condition}.",
                "For {condition}, standard treatments consist of {treatments}.",
                "Managing {condition} typically involves {treatments}.",
                "The medical approach to treating {condition} includes {treatments}."
            ],
            "causes": [
                "The main causes of {condition} are {causes}.",
                "{condition} is typically caused by {causes}.",
                "Factors that contribute to developing {condition} include {causes}.",
                "Research suggests {condition} develops due to {causes}.",
                "The etiology of {condition} involves {causes}."
            ],
            "prevention": [
                "To help prevent {condition}, it's recommended to {prevention}.",
                "Preventative measures for {condition} include {prevention}.",
                "You may reduce your risk of {condition} by {prevention}.",
                "Dermatologists advise {prevention} to avoid {condition}.",
                "Strategies to prevent {condition} focus on {prevention}."
            ],
            "comparison": [
                "When comparing {condition1} and {condition2}, the main differences are in {differences}.",
                "{condition1} and {condition2} differ primarily in {differences}.",
                "While both are skin conditions, {condition1} and {condition2} can be distinguished by {differences}.",
                "The key distinction between {condition1} and {condition2} is {differences}.",
                "Dermatologists differentiate {condition1} from {condition2} by looking at {differences}."
            ],
            "unknown": [
                "I don't have specific information about that skin condition.",
                "That particular skin condition isn't in my knowledge base.",
                "I'm not familiar with details about that specific condition.",
                "I don't have enough information to provide details about that condition.",
                "That's beyond the scope of my current knowledge about skin conditions."
            ],
            "general": [
                "When it comes to skin health, {general_advice}.",
                "Dermatologists generally recommend that {general_advice}.",
                "For overall skin health, it's important to {general_advice}.",
                "A good approach to skin care includes {general_advice}.",
                "Medical experts suggest {general_advice} for maintaining healthy skin."
            ]
        }
        
        # General skin health advice for fallback responses
        self.general_skin_advice = [
            "protect your skin from the sun by using broad-spectrum sunscreen",
            "stay hydrated and maintain a balanced diet rich in antioxidants",
            "avoid smoking and limit alcohol consumption for better skin health",
            "establish a consistent skincare routine appropriate for your skin type",
            "get regular skin checks, especially if you have risk factors for skin cancer",
            "manage stress through healthy coping mechanisms to prevent stress-related skin issues",
            "be gentle with your skin, avoiding harsh products and excessive washing",
            "pay attention to new or changing skin lesions and consult a dermatologist promptly",
            "consider environmental factors like humidity and temperature that may affect your skin",
            "don't share personal items like towels or makeup to prevent spreading skin infections"
        ]
    
    def _initialize_language_patterns(self):
        """Initialize language patterns for query understanding"""
        # Query type recognition patterns
        self.query_patterns = {
            "description": [
                r"what (?:is|are) (\w+\s*\w*)",
                r"tell me about (\w+\s*\w*)",
                r"describe (\w+\s*\w*)",
                r"information on (\w+\s*\w*)",
                r"explain (\w+\s*\w*)"
            ],
            "symptoms": [
                r"symptoms of (\w+\s*\w*)",
                r"signs of (\w+\s*\w*)",
                r"how to identify (\w+\s*\w*)",
                r"how do I know if I have (\w+\s*\w*)",
                r"what does (\w+\s*\w*) look like"
            ],
            "treatments": [
                r"how to treat (\w+\s*\w*)",
                r"treatment for (\w+\s*\w*)",
                r"how to manage (\w+\s*\w*)",
                r"cure for (\w+\s*\w*)",
                r"medication for (\w+\s*\w*)",
                r"therapy for (\w+\s*\w*)"
            ],
            "causes": [
                r"what causes (\w+\s*\w*)",
                r"why do people get (\w+\s*\w*)",
                r"risk factors for (\w+\s*\w*)",
                r"how do you develop (\w+\s*\w*)",
                r"etiology of (\w+\s*\w*)"
            ],
            "prevention": [
                r"how to prevent (\w+\s*\w*)",
                r"preventing (\w+\s*\w*)",
                r"avoid (\w+\s*\w*)",
                r"reduce risk of (\w+\s*\w*)",
                r"protection from (\w+\s*\w*)"
            ],
            "comparison": [
                r"difference between (\w+\s*\w*) and (\w+\s*\w*)",
                r"compare (\w+\s*\w*) and (\w+\s*\w*)",
                r"(\w+\s*\w*) vs (\w+\s*\w*)",
                r"distinguish (\w+\s*\w*) from (\w+\s*\w*)",
                r"how are (\w+\s*\w*) and (\w+\s*\w*) different"
            ]
        }
        
        # Condition name variations and synonyms for better matching
        self.condition_synonyms = {
            "acne": ["pimples", "zits", "breakouts", "acne vulgaris", "cystic acne", "acne rosacea"],
            "hyperpigmentation": ["dark spots", "sun spots", "age spots", "melasma", "brown spots", "skin discoloration"],
            "nail_psoriasis": ["nail psoriasis", "psoriatic nails", "psoriasis of nails", "psoriatic nail disease"],
            "sjs_ten": ["stevens johnson syndrome", "toxic epidermal necrolysis", "sjs", "ten", "stevens-johnson syndrome", "stevens johnson", "sjs/ten"],
            "vitiligo": ["leucoderma", "white patches", "skin depigmentation", "vitiligo vulgaris"],
            "eczema": ["atopic dermatitis", "dermatitis", "atopic eczema", "nummular eczema", "dyshidrotic eczema"],
            "rosacea": ["acne rosacea", "adult acne", "facial redness", "rhinophyma"],
            "psoriasis": ["plaque psoriasis", "scalp psoriasis", "psoriatic disease", "pustular psoriasis", "guttate psoriasis"],
            "fungal_infections": ["ringworm", "athlete's foot", "jock itch", "tinea", "candidiasis", "fungal rash", "dermatophytosis", "onychomycosis"],
            "warts": ["verruca", "plantar warts", "genital warts", "common warts", "flat warts", "filiform warts"]
        }
    
    def _identify_condition(self, query: str) -> str:
        """
        Identify the skin condition being asked about in the query.
        
        Args:
            query: The user's query text
            
        Returns:
            The identified condition key or None if not found
        """
        # First try direct matching with knowledge graph keys
        query_lower = query.lower()
        for condition in self.knowledge_graph.keys():
            if condition.lower() in query_lower:
                return condition
        
        # Try matching with condition synonyms
        for condition, synonyms in self.condition_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    return condition
        
        # Try to identify based on symptom matching
        for symptom, conditions in self.symptom_to_condition.items():
            if symptom.lower() in query_lower:
                if conditions:
                    # Return the first associated condition
                    return conditions[0]
        
        # No match found
        return None
    
    def _identify_query_type(self, query: str) -> str:
        """
        Identify the type of query being asked.
        
        Args:
            query: The user's query text
            
        Returns:
            The query type (description, symptoms, treatments, etc.)
        """
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return query_type
        
        # Default to description if no specific type is identified
        return "description"
    
    def _extract_comparison_conditions(self, query: str) -> Tuple[str, str]:
        """
        Extract two conditions being compared in a comparison query.
        
        Args:
            query: The user's query text
            
        Returns:
            Tuple of (condition1, condition2) or (None, None) if not found
        """
        query_lower = query.lower()
        
        for pattern in self.query_patterns["comparison"]:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match and len(match.groups()) >= 2:
                cond1_text = match.group(1)
                cond2_text = match.group(2)
                
                # Try to map to known conditions
                cond1 = self._identify_condition(cond1_text)
                cond2 = self._identify_condition(cond2_text)
                
                if cond1 and cond2:
                    return (cond1, cond2)
        
        return (None, None)
    
    def _format_list(self, items: List[str], limit: int = 5) -> str:
        """Format a list of items into a natural language string with limit"""
        if not items:
            return ""
        
        limited_items = items[:limit]
        
        if len(limited_items) == 1:
            return limited_items[0]
        elif len(limited_items) == 2:
            return f"{limited_items[0]} and {limited_items[1]}"
        else:
            return ", ".join(limited_items[:-1]) + f", and {limited_items[-1]}"
    
    def _compare_conditions(self, cond1: str, cond2: str) -> str:
        """Generate a comparison between two skin conditions"""
        if not (cond1 in self.knowledge_graph and cond2 in self.knowledge_graph):
            return "I don't have enough information to compare these conditions."
        
        c1 = self.knowledge_graph[cond1]
        c2 = self.knowledge_graph[cond2]
        
        differences = []
        
        # Compare causes
        if 'causes' in c1 and 'causes' in c2:
            c1_causes = self._format_list(c1['causes'], 3)
            c2_causes = self._format_list(c2['causes'], 3)
            differences.append(f"{cond1} is typically caused by {c1_causes}, while {cond2} is caused by {c2_causes}")
        
        # Compare symptoms
        if 'symptoms' in c1 and 'symptoms' in c2:
            c1_symptoms = self._format_list(c1['symptoms'], 3)
            c2_symptoms = self._format_list(c2['symptoms'], 3)
            differences.append(f"{cond1} presents with {c1_symptoms}, whereas {cond2} shows {c2_symptoms}")
        
        # Compare treatments
        if 'treatments' in c1 and 'treatments' in c2:
            c1_treatments = self._format_list(c1['treatments'], 3)
            c2_treatments = self._format_list(c2['treatments'], 3)
            differences.append(f"{cond1} is typically treated with {c1_treatments}, while {cond2} is managed with {c2_treatments}")
        
        # Compare affected areas
        if 'affected_areas' in c1 and 'affected_areas' in c2:
            c1_areas = self._format_list(c1['affected_areas'], 3)
            c2_areas = self._format_list(c2['affected_areas'], 3)
            differences.append(f"{cond1} commonly affects {c1_areas}, while {cond2} typically involves {c2_areas}")
        
        if not differences:
            return f"Both {cond1} and {cond2} are skin conditions, but I don't have enough comparative information."
        
        return "\n\n".join(differences)
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        Generate a response to the given prompt using the knowledge base.
        
        Args:
            prompt: The user's query
            max_tokens: Maximum response length (unused but kept for API compatibility)
            temperature: Randomness parameter (used for template selection)
            
        Returns:
            Generated response text
        """
        if not self.model_loaded:
            return "Model not loaded. Please initialize the model first."
        
        # Identify the query type and condition
        query_type = self._identify_query_type(prompt)
        
        if query_type == "comparison":
            cond1, cond2 = self._extract_comparison_conditions(prompt)
            if cond1 and cond2:
                return self._compare_conditions(cond1, cond2)
            else:
                return "I couldn't identify the conditions you're trying to compare. Could you please specify the exact skin conditions?"
        
        condition = self._identify_condition(prompt)
        
        if not condition:
            # Fallback for unrecognized conditions
            template_idx = min(int(temperature * 5), 4)  # Use temperature to select template
            general_template = self.response_templates["general"][template_idx]
            advice_idx = min(int(temperature * 10), 9)  # Use temperature to select advice
            general_advice = self.general_skin_advice[advice_idx]
            
            return general_template.format(general_advice=general_advice)
        
        # Get relevant information from knowledge graph
        if condition not in self.knowledge_graph:
            template_idx = min(int(temperature * 5), 4)
            return self.response_templates["unknown"][template_idx]
        
        condition_info = self.knowledge_graph[condition]
        
        # Handle specific query types
        if query_type == "description" and "description" in condition_info:
            template_idx = min(int(temperature * len(self.response_templates["description"])), len(self.response_templates["description"]) - 1)
            template = self.response_templates["description"][template_idx]
            return template.format(condition=condition, description=condition_info["description"])
        
        elif query_type == "symptoms" and "symptoms" in condition_info:
            template_idx = min(int(temperature * len(self.response_templates["symptoms"])), len(self.response_templates["symptoms"]) - 1)
            template = self.response_templates["symptoms"][template_idx]
            symptoms = self._format_list(condition_info["symptoms"])
            return template.format(condition=condition, symptoms=symptoms)
        
        elif query_type == "treatments" and "treatments" in condition_info:
            template_idx = min(int(temperature * len(self.response_templates["treatments"])), len(self.response_templates["treatments"]) - 1)
            template = self.response_templates["treatments"][template_idx]
            treatments = self._format_list(condition_info["treatments"])
            return template.format(condition=condition, treatments=treatments)
        
        elif query_type == "causes" and "causes" in condition_info:
            template_idx = min(int(temperature * len(self.response_templates["causes"])), len(self.response_templates["causes"]) - 1)
            template = self.response_templates["causes"][template_idx]
            causes = self._format_list(condition_info["causes"])
            return template.format(condition=condition, causes=causes)
        
        elif query_type == "prevention" and "prevention" in condition_info:
            template_idx = min(int(temperature * len(self.response_templates["prevention"])), len(self.response_templates["prevention"]) - 1)
            template = self.response_templates["prevention"][template_idx]
            prevention = self._format_list(condition_info["prevention"])
            return template.format(condition=condition, prevention=prevention)
        
        # Fallback to description if specific query type not available
        if "description" in condition_info:
            template_idx = min(int(temperature * len(self.response_templates["description"])), len(self.response_templates["description"]) - 1)
            template = self.response_templates["description"][template_idx]
            return template.format(condition=condition, description=condition_info["description"])
        
        # Ultimate fallback
        return f"I have information about {condition}, but not the specific details you're asking for."
    
    def generate_stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Stream the generated response token by token.
        
        Args:
            prompt: The user's query
            max_tokens: Maximum response length (unused but kept for API compatibility)
            temperature: Randomness parameter
            
        Yields:
            Generated text tokens one by one
        """
        # Generate the full response first
        full_response = self.generate(prompt, max_tokens, temperature)
        
        # Simulate streaming by yielding one word at a time
        words = full_response.split()
        for i, word in enumerate(words):
            # Add space before word (except for first word)
            if i > 0:
                yield " "
            
            # For extra realism, stream long words character by character
            if len(word) > 7:
                for char in word:
                    yield char
                    time.sleep(0.01)  # Small delay between characters
            else:
                yield word
            
            # Small random delay between words for realistic typing effect
            time.sleep(0.03 + random.random() * 0.07)
    
    def unload(self):
        """Unload the model from memory"""
        self.model_loaded = False

class EnhancedLLMManager:
    """
    Manager class for handling the EnhancedSkinLLM model lifecycle and inference.
    """
    
    def __init__(self, model_path=None, context_size=512, threads=4):
        """
        Initialize the LLM manager.
        
        Args:
            model_path: Path to the knowledge base files (if any)
            context_size: Maximum context size for the model
            threads: Number of threads for parallel processing
        """
        self.model_path = model_path
        self.context_size = context_size
        self.threads = threads
        self.model = None
        self.model_loaded = False
        
        # Statistics tracking
        self.total_requests = 0
        self.avg_response_time = 0
        self.last_response_time = 0
        
        # Message queue for background processing
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.worker_thread = None
        self.worker_running = False
    
    def load_model(self):
        """Load the model if not already loaded"""
        if not self.model_loaded:
            print("Loading EnhancedSkinLLM model...")
            self.model = EnhancedSkinLLM(
                model_path=self.model_path,
                context_size=self.context_size,
                threads=self.threads
            )
            self.model_loaded = True
            print("EnhancedSkinLLM model loaded successfully.")
    
    def unload_model(self):
        """Unload the model from memory"""
        if self.model_loaded and self.model:
            self.model.unload()
            self.model = None
            self.model_loaded = False
            print("Model unloaded from memory.")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input text to generate from
            
        Returns:
            Generated text response
        """
        if not self.model_loaded:
            self.load_model()
        
        self.total_requests += 1
        start_time = time.time()
        
        response = self.model.generate(prompt)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Update statistics
        self.last_response_time = response_time
        self.avg_response_time = (self.avg_response_time * (self.total_requests - 1) + response_time) / self.total_requests
        
        print(f"Response generated in {response_time:.2f} seconds")
        
        return response
    
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream the generated response token by token.
        
        Args:
            prompt: Input text to generate from
            
        Yields:
            Generated text tokens
        """
        if not self.model_loaded:
            self.load_model()
        
        self.total_requests += 1
        start_time = time.time()
        
        # Yield tokens from the model
        for token in self.model.generate_stream(prompt):
            yield token
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Update statistics
        self.last_response_time = response_time
        self.avg_response_time = (self.avg_response_time * (self.total_requests - 1) + response_time) / self.total_requests
        
        print(f"Response generated in {response_time:.2f} seconds")
    
    def start_worker(self):
        """Start the worker thread for processing messages asynchronously"""
        if not self.worker_thread or not self.worker_thread.is_alive():
            self.worker_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            print("Worker thread started")
    
    def stop_worker(self):
        """Stop the worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_running = False
            self.worker_thread.join(timeout=1.0)
            print("Worker thread stopped")
    
    def _worker_loop(self):
        """Worker thread function for processing messages in the background"""
        while self.worker_running:
            try:
                # Get a message from the queue with timeout
                message = self.message_queue.get(timeout=0.1)
                
                # Process the message
                response = self.generate_response(message)
                
                # Put the response in the response queue
                self.response_queue.put(response)
                
                # Mark the task as done
                self.message_queue.task_done()
            except queue.Empty:
                # Queue was empty, continue waiting
                pass
            except Exception as e:
                print(f"Error in worker thread: {e}")
                # Put the error message in the response queue
                self.response_queue.put(f"Error: {str(e)}")
                
                try:
                    # Mark the task as done if there was a message
                    self.message_queue.task_done()
                except:
                    pass
    
    def submit_message(self, message: str):
        """
        Submit a message for asynchronous processing.
        
        Args:
            message: The message to process
        """
        if not self.worker_running:
            self.start_worker()
        
        self.message_queue.put(message)
        print(f"Message submitted (queue size: {self.message_queue.qsize()})")
    
    def get_response(self, block=True, timeout=None) -> str:
        """
        Get a response from the response queue.
        
        Args:
            block: Whether to block until a response is available
            timeout: Timeout in seconds
            
        Returns:
            Response string or None if timeout
        """
        try:
            response = self.response_queue.get(block=block, timeout=timeout)
            print(f"Response retrieved (len: {len(response) if response else 0})")
            return response
        except queue.Empty:
            print("Response timeout")
            return "I apologize for the delay. I'm still processing your request."
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the LLM usage
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_requests": self.total_requests,
            "avg_response_time": self.avg_response_time,
            "last_response_time": self.last_response_time,
            "model_loaded": self.model_loaded,
            "queue_size": self.message_queue.qsize() if hasattr(self.message_queue, 'qsize') else 0
        }
    
    def __del__(self):
        """Cleanup when the object is deleted"""
        print("EnhancedLLMManager cleanup")
        self.stop_worker()
        self.unload_model()