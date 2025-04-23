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

class SkinLLM:
    """
    A real-time, knowledge-based, and rule-based LLM for skin disease information.
    This implementation does not use transformers but instead uses pattern matching,
    knowledge graphs, and statistical methods to generate responses.
    """
    
    def __init__(self, model_path=None, context_size=512, threads=4):
        """
        Initialize the SkinLLM model.
        
        Args:
            model_path: Path to knowledge base files (if any)
            context_size: Memory context size
            threads: Number of threads for parallel processing
        """
        self.context_size = context_size
        self.threads = threads
        self.model_loaded = False
        
        # Knowledge graph for skin conditions
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
                "complications": ["Psychological distress", "Self-esteem issues"],
                "affected_areas": ["Face", "Hands", "Any area exposed to sun", "Previously inflamed areas"],
                "statistics": "Melasma affects up to 6 million women in the United States.",
                "related_conditions": ["Vitiligo", "Albinism", "Addison's disease"]
            },
            "nail psoriasis": {
                "description": "A manifestation of psoriasis affecting the fingernails and toenails.",
                "symptoms": ["Pitting", "Onycholysis", "Oil spots", "Subungual hyperkeratosis", "Splinter hemorrhages", "Crumbling", "Discoloration", "Ridges"],
                "treatments": ["Topical steroids", "Vitamin D analogues", "Systemic medications", "Biologics", "Intralesional injections", "Phototherapy"],
                "prevention": ["Avoiding nail trauma", "Proper nail care", "Treating underlying psoriasis", "Avoiding infections", "Keeping nails short"],
                "causes": ["Autoimmune factors", "Genetics", "Environmental triggers"],
                "complications": ["Functional impairment", "Pain", "Secondary infections", "Psychological distress"],
                "affected_areas": ["Fingernails", "Toenails", "Nail bed", "Nail matrix"],
                "statistics": "Affects about 50% of people with psoriasis and 80% of those with psoriatic arthritis.",
                "related_conditions": ["Plaque psoriasis", "Psoriatic arthritis", "Fungal nail infections"]
            },
            "sjs-ten": {
                "description": "Stevens-Johnson syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) are severe, life-threatening skin reactions, usually to medications.",
                "symptoms": ["Skin pain", "Widespread rash", "Blisters", "Peeling skin", "Fever", "Fatigue", "Burning eyes", "Mucosal involvement"],
                "treatments": ["Stopping causative medication", "Supportive care", "Corticosteroids", "Intravenous immunoglobulin", "Hospitalization", "Wound care", "Pain management"],
                "prevention": ["Avoiding triggering medications", "Genetic testing", "Medical alert bracelet", "Immediate medical attention", "Careful medication monitoring"],
                "causes": ["Medication reactions", "Infections", "Genetic factors", "Vaccination (rare)"],
                "complications": ["Sepsis", "Shock", "Multi-organ failure", "Permanent skin damage", "Visual impairment", "Death"],
                "affected_areas": ["Skin (widespread)", "Mucous membranes", "Eyes", "Mouth", "Genitals"],
                "statistics": "SJS affects 1-2 persons per million annually. TEN is even rarer with 0.4-1.2 cases per million annually.",
                "related_conditions": ["Erythema multiforme", "Drug reactions", "Pemphigus"]
            },
            "vitiligo": {
                "description": "A disease that causes loss of skin color in patches due to destruction of pigment cells (melanocytes).",
                "symptoms": ["White patches on skin", "Premature whitening of hair", "Loss of color in tissues inside mouth", "Change in eye color", "Smooth texture of white patches"],
                "treatments": ["Topical corticosteroids", "Calcineurin inhibitors", "Phototherapy", "Skin grafting", "Depigmentation", "Micropigmentation", "Monobenzone"],
                "prevention": ["Sun protection", "Stress management", "Avoiding skin trauma", "Early treatment", "Healthy lifestyle"],
                "causes": ["Autoimmune disorder", "Genetic factors", "Neurogenic factors", "Oxidative stress", "Environmental triggers"],
                "complications": ["Sunburn", "Psychological impact", "Social stigma", "Increased risk of skin cancer in depigmented areas"],
                "affected_areas": ["Face", "Hands", "Arms", "Feet", "Genital areas", "Any part of body with pigment"],
                "statistics": "Affects approximately 1% of the world's population regardless of race or ethnicity.",
                "related_conditions": ["Autoimmune thyroid diseases", "Addison's disease", "Pernicious anemia", "Alopecia areata"]
            }
        }
        
        # Advanced knowledge for complex queries
        self.advanced_knowledge = {
            "differential_diagnosis": {
                "red_patches": ["Eczema", "Psoriasis", "Contact dermatitis", "Rosacea", "Fungal infections"],
                "white_patches": ["Vitiligo", "Pityriasis versicolor", "Tuberous sclerosis", "Idiopathic guttate hypomelanosis"],
                "itchy_skin": ["Eczema", "Allergic reactions", "Scabies", "Psoriasis", "Hives", "Xerosis (dry skin)"],
                "raised_bumps": ["Acne", "Folliculitis", "Keratosis pilaris", "Warts", "Molluscum contagiosum"]
            },
            "treatment_cautions": {
                "retinoids": "Avoid during pregnancy. Can cause dryness, redness, and increased sun sensitivity.",
                "corticosteroids": "Long-term use can lead to skin thinning, stretch marks, and adrenal suppression.",
                "biologics": "May increase risk of infections and require monitoring for tuberculosis and other diseases.",
                "antibiotics": "Can lead to antibiotic resistance if used improperly. May cause digestive issues.",
                "hydroquinone": "Should not be used for more than 3 months continuously due to risk of ochronosis."
            },
            "emerging_treatments": {
                "acne": ["Nitric oxide-releasing topicals", "Cannabidiol (CBD) products", "Systemic clascoterone", "Topical minocycline"],
                "psoriasis": ["IL-23 inhibitors", "TYK2 inhibitors", "JAK inhibitors", "Oral minoxidil"],
                "vitiligo": ["JAK inhibitors", "Mini-grafting procedures", "Cellular transplantation"],
                "hyperpigmentation": ["Tranexamic acid", "Cysteamine cream", "Glutathione treatments", "Microbiome-modulating agents"]
            }
        }
        
        # Statistical NLU (Natural Language Understanding) patterns
        self.query_patterns = {
            "symptoms": [r"symptoms\s+of\s+(.*)", r"signs\s+of\s+(.*)", r"how.*identify\s+(.*)", r"what.*look\s+like\s+(.*)", r"how.*know.*have\s+(.*)"],
            "treatments": [r"treat\s+(.*)", r"cure\s+(.*)", r"heal\s+(.*)", r"medication\s+for\s+(.*)", r"therapy\s+for\s+(.*)", r"remedy\s+for\s+(.*)"],
            "prevention": [r"prevent\s+(.*)", r"avoid\s+(.*)", r"reduce.*risk\s+of\s+(.*)", r"protect.*from\s+(.*)", r"stop\s+(.*)\s+from"],
            "causes": [r"cause\s+of\s+(.*)", r"why.*get\s+(.*)", r"what.*lead\s+to\s+(.*)", r"reason\s+for\s+(.*)", r"factor.*for\s+(.*)"],
            "comparison": [r"difference\s+between\s+(.*)\s+and\s+(.*)", r"compare\s+(.*)\s+and\s+(.*)", r"(.*)\s+versus\s+(.*)", r"(.*)\s+vs\s+(.*)"],
            "general": [r"what\s+is\s+(.*)", r"information\s+about\s+(.*)", r"tell.*about\s+(.*)", r"explain\s+(.*)", r"describe\s+(.*)"],
            "time": [r"how\s+long\s+(.*)", r"duration\s+of\s+(.*)", r"time.*to\s+(.*)", r"when\s+will\s+(.*)"],
            "severity": [r"how\s+serious\s+is\s+(.*)", r"is\s+(.*)\s+dangerous", r"risk\s+of\s+(.*)", r"complication.*of\s+(.*)"]
        }
        
        # TF-IDF like features for improved matching
        self._calculate_keyword_weights()
        
        # Response templates for natural-sounding variation
        self.response_templates = {
            "symptoms": [
                "Common symptoms of {condition} include: {symptoms}.",
                "People with {condition} often experience: {symptoms}.",
                "{condition} typically presents with these signs: {symptoms}.",
                "The main symptoms to look for with {condition} are: {symptoms}.",
                "If you have {condition}, you might notice: {symptoms}."
            ],
            "treatments": [
                "Common treatments for {condition} include: {treatments}.",
                "{condition} is typically treated with: {treatments}.",
                "Healthcare providers often recommend these treatments for {condition}: {treatments}.",
                "Effective treatment options for {condition} include: {treatments}.",
                "To treat {condition}, doctors may prescribe or recommend: {treatments}."
            ],
            "prevention": [
                "To help prevent {condition}, try to: {prevention}.",
                "These strategies may help reduce your risk of {condition}: {prevention}.",
                "For {condition} prevention, experts recommend: {prevention}.",
                "You can take these steps to help avoid {condition}: {prevention}.",
                "Preventive measures for {condition} include: {prevention}."
            ],
            "description": [
                "{condition} is {description}",
                "{description} This condition is known as {condition}.",
                "{condition}, which {description}",
                "As a skin condition, {condition} {description}",
                "Medically speaking, {condition} refers to {description}"
            ],
            "not_found": [
                "I don't have specific information about that condition. I can provide details about acne, hyperpigmentation, nail psoriasis, SJS-TEN, and vitiligo.",
                "I'm not familiar with that specific condition. Would you like information about one of these: acne, hyperpigmentation, nail psoriasis, SJS-TEN, or vitiligo?",
                "That's outside my current knowledge base. I can tell you about acne, hyperpigmentation, nail psoriasis, SJS-TEN, and vitiligo if you're interested.",
                "I don't have data on that particular condition. I'm best equipped to discuss acne, hyperpigmentation, nail psoriasis, SJS-TEN, and vitiligo.",
                "I haven't been trained on that specific condition. Would you like to know about acne, hyperpigmentation, nail psoriasis, SJS-TEN, or vitiligo instead?"
            ]
        }
        
        # Named entity recognition for extracting conditions from text
        self.condition_aliases = {
            "acne": ["acne", "pimples", "zits", "breakouts", "cystic acne", "acne vulgaris", "blackheads", "whiteheads"],
            "hyperpigmentation": ["hyperpigmentation", "dark spots", "age spots", "sun spots", "melasma", "chloasma", "liver spots", "post-inflammatory hyperpigmentation", "pih"],
            "nail psoriasis": ["nail psoriasis", "psoriatic nails", "psoriasis of the nails", "nail disease", "nail condition", "psoriatic nail disease"],
            "sjs-ten": ["sjs-ten", "stevens-johnson syndrome", "toxic epidermal necrolysis", "sjs", "ten", "stevens johnson", "drug reaction"],
            "vitiligo": ["vitiligo", "white patches", "depigmentation", "leucoderma", "skin depigmentation", "loss of skin color"]
        }
        
        # Conversation memory for contextual awareness
        self.conversation_memory = []
        self.memory_size = 5  # Remember last 5 exchanges
        
        # Load the model - in this case, initialize our token bank from the knowledge graph
        print("Loading SkinLLM model...")
        # Create token bank for more natural language generation
        self._initialize_token_bank()
        self.model_loaded = True
        print("SkinLLM model loaded successfully!")
    
    def _initialize_token_bank(self):
        """Initialize token bank for language generation from knowledge graph"""
        self.token_bank = {}
        
        # Process all text in the knowledge graph to build a simple Markov chain
        all_text = []
        
        for condition, data in self.knowledge_graph.items():
            all_text.append(data["description"])
            all_text.extend(data["symptoms"])
            all_text.extend(data["treatments"]) 
            all_text.extend(data["prevention"])
            
            if "causes" in data:
                all_text.extend(data["causes"])
            if "complications" in data:
                all_text.extend(data["complications"])
            if "statistics" in data:
                all_text.append(data["statistics"])
        
        # Create token sequences
        for text in all_text:
            words = text.lower().split()
            if len(words) < 2:
                continue
                
            for i in range(len(words) - 1):
                if words[i] not in self.token_bank:
                    self.token_bank[words[i]] = []
                self.token_bank[words[i]].append(words[i+1])
    
    def _calculate_keyword_weights(self):
        """Calculate TF-IDF like weights for important keywords"""
        self.keyword_weights = {}
        
        # Count occurrences of words across all conditions
        word_counts = {}
        total_documents = len(self.knowledge_graph)
        
        for condition, data in self.knowledge_graph.items():
            # Combine all text data for this condition
            all_text = data["description"] + " " + " ".join(data["symptoms"]) + " " + " ".join(data["treatments"]) + " " + " ".join(data["prevention"])
            
            # Extract words
            words = re.findall(r'\b\w+\b', all_text.lower())
            
            # Count unique words in this document
            unique_words = set(words)
            for word in unique_words:
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
        
        # Calculate IDF for all words
        for word, count in word_counts.items():
            # Simple IDF calculation
            idf = math.log(total_documents / count)
            
            # Store in keyword weights
            self.keyword_weights[word] = idf
    
    def _identify_condition(self, text):
        """
        Identify which skin condition is being asked about.
        Uses a combination of direct matching and semantic similarity.
        
        Args:
            text: The query text
            
        Returns:
            Identified condition or None
        """
        text_lower = text.lower()
        
        # Try direct matching with condition names and aliases
        for condition, aliases in self.condition_aliases.items():
            for alias in aliases:
                if alias in text_lower:
                    return condition
        
        # Look for partial matches for more complex queries
        for condition in self.knowledge_graph:
            # Get keywords for this condition
            keywords = set()
            for alias in self.condition_aliases[condition]:
                keywords.update(alias.lower().split())
            
            # Check for matches
            matching_words = sum(1 for word in keywords if word in text_lower)
            if matching_words >= 2 or (len(keywords) == 1 and matching_words == 1):
                return condition
        
        # If still not identified, try looking at conversation memory for context
        if self.conversation_memory:
            last_query, last_response = self.conversation_memory[-1]
            for condition in self.knowledge_graph:
                if condition in last_response.lower():
                    # If previous response mentioned a condition, user might be following up
                    return condition
                
        return None
    
    def _identify_query_type(self, text):
        """
        Identify what type of query is being asked (symptoms, treatments, etc.)
        
        Args:
            text: The query text
            
        Returns:
            Query type or None
        """
        text_lower = text.lower()
        
        # Check for direct question types
        query_type_keywords = {
            "symptoms": ["symptoms", "signs", "look like", "identifying", "recognize", "how do i know"],
            "treatments": ["treatment", "cure", "therapy", "manage", "heal", "get rid of", "medication", "remedy"],
            "prevention": ["prevent", "avoid", "reduce risk", "protecting from", "stop from getting"],
            "causes": ["cause", "reason", "why", "what leads to", "factors", "how do you get", "where does"],
            "description": ["what is", "tell me about", "explain", "describe", "information on"],
            "time": ["how long", "duration", "time to", "when will", "timeline"],
            "severity": ["how serious", "dangerous", "life-threatening", "risk", "complication", "fatal", "death"]
        }
        
        # Check each query type
        for query_type, keywords in query_type_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return query_type
        
        # Check patterns with regex for more complex queries
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return query_type
        
        # If type still not identified, default to "description" for general information
        return "description"
    
    def _extract_comparison_conditions(self, text):
        """
        Extract two conditions being compared in a query
        
        Args:
            text: The query text
            
        Returns:
            Tuple of two conditions or None
        """
        text_lower = text.lower()
        
        # Check comparison patterns
        for pattern in self.query_patterns["comparison"]:
            match = re.search(pattern, text_lower)
            if match and len(match.groups()) >= 2:
                condition1 = match.group(1).strip()
                condition2 = match.group(2).strip()
                
                # Resolve to known conditions
                resolved1 = None
                resolved2 = None
                
                for cond, aliases in self.condition_aliases.items():
                    for alias in aliases:
                        if alias in condition1:
                            resolved1 = cond
                        if alias in condition2:
                            resolved2 = cond
                
                if resolved1 and resolved2:
                    return (resolved1, resolved2)
        
        return None
    
    def _get_response_template(self, query_type):
        """Get a random response template for the query type"""
        if query_type in self.response_templates:
            return random.choice(self.response_templates[query_type])
        return random.choice(self.response_templates["description"])
    
    def _format_list(self, items, limit=5):
        """Format a list of items into a natural language string with limit"""
        if not items:
            return "no specific information available"
            
        # Limit the number of items if necessary
        if len(items) > limit:
            items = items[:limit]
            items[-1] = items[-1] + ", and others"
        
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        Generate a response based on the given prompt.
        
        Args:
            prompt: Input text to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Generated text
        """
        # Simulate model thinking time - this is real processing time
        start_time = time.time()
        
        # Process the prompt to understand the query
        prompt_lower = prompt.lower()
        
        # Add general greeting responses
        if re.search(r'\b(hi|hello|hey|greetings)\b', prompt_lower) and len(prompt_lower.split()) < 5:
            return "Hello! I'm your skin disease information assistant. I can provide information about conditions like acne, hyperpigmentation, nail psoriasis, SJS-TEN, and vitiligo. How can I help you today?"
        
        if re.search(r'\b(how are you|how do you feel|how\'s it going)\b', prompt_lower) and len(prompt_lower.split()) < 7:
            return "I'm functioning well, thank you! I'm here to provide information about skin conditions. What would you like to know about?"
        
        # Check if the user wants a comparison between conditions
        comparison = self._extract_comparison_conditions(prompt)
        if comparison:
            condition1, condition2 = comparison
            if condition1 in self.knowledge_graph and condition2 in self.knowledge_graph:
                diff_response = f"When comparing {condition1} and {condition2}:\n\n"
                
                # Compare key aspects
                aspects = ["description", "symptoms", "treatments", "causes"]
                for aspect in aspects:
                    diff_response += f"• {aspect.capitalize()}: \n"
                    diff_response += f"  - {condition1.capitalize()}: "
                    if aspect == "description":
                        diff_response += f"{self.knowledge_graph[condition1][aspect]}\n"
                    else:
                        items = self.knowledge_graph[condition1].get(aspect, ["No information available"])
                        diff_response += f"{', '.join(items[:3])}\n"
                    
                    diff_response += f"  - {condition2.capitalize()}: "
                    if aspect == "description":
                        diff_response += f"{self.knowledge_graph[condition2][aspect]}\n"
                    else:
                        items = self.knowledge_graph[condition2].get(aspect, ["No information available"])
                        diff_response += f"{', '.join(items[:3])}\n"
                
                return diff_response
        
        # Identify the skin condition being asked about
        condition = self._identify_condition(prompt)
        
        # Get the type of query (symptoms, treatments, etc.)
        query_type = self._identify_query_type(prompt)
        
        # Generate response based on identified condition and query type
        if condition and condition in self.knowledge_graph:
            condition_data = self.knowledge_graph[condition]
            
            if query_type == "symptoms":
                symptoms = self._format_list(condition_data["symptoms"])
                template = self._get_response_template("symptoms")
                response = template.format(condition=condition.capitalize(), symptoms=symptoms)
                
            elif query_type == "treatments":
                treatments = self._format_list(condition_data["treatments"])
                template = self._get_response_template("treatments")
                response = template.format(condition=condition.capitalize(), treatments=treatments)
                
            elif query_type == "prevention":
                prevention = self._format_list(condition_data["prevention"])
                template = self._get_response_template("prevention")
                response = template.format(condition=condition.capitalize(), prevention=prevention)
                
            elif query_type == "causes" and "causes" in condition_data:
                causes = self._format_list(condition_data["causes"])
                response = f"The main causes of {condition.capitalize()} include: {causes}."
                
            elif query_type == "time":
                if condition == "acne":
                    response = "Acne treatments typically take 4-8 weeks to show improvement. Complete resolution may take several months of consistent treatment."
                elif condition == "hyperpigmentation":
                    response = "Treatment for hyperpigmentation usually takes 3-6 months. Older or deeper pigmentation may take longer to fade."
                elif condition == "nail psoriasis":
                    response = "Nail psoriasis treatment may take 6-12 months since nails grow slowly. Complete clearing may not be achieved in all cases."
                elif condition == "sjs-ten":
                    response = "SJS-TEN requires immediate medical attention. Recovery typically takes 2-6 weeks, but complete healing may take months. Long-term complications may persist."
                elif condition == "vitiligo":
                    response = "Vitiligo treatment is typically long-term. Results from treatment may take 3-12 months to become noticeable. Some areas may not repigment."
                else:
                    response = f"Treatment duration for {condition} varies depending on severity, treatment method, and individual response to treatment."
            
            elif query_type == "severity":
                if condition == "acne":
                    response = "Acne severity ranges from mild (few pimples) to severe (cystic acne). While not life-threatening, severe acne can cause significant scarring and psychological distress."
                elif condition == "hyperpigmentation":
                    response = "Hyperpigmentation is generally not dangerous to physical health, but it can be psychologically distressing. It's primarily a cosmetic concern."
                elif condition == "nail psoriasis":
                    response = "Nail psoriasis itself is not dangerous but can be painful and affect daily activities. It can also indicate more severe psoriasis elsewhere or psoriatic arthritis."
                elif condition == "sjs-ten":
                    response = "SJS-TEN is a medical emergency with mortality rates of 10% for SJS and up to 30% for TEN. Immediate hospitalization is essential."
                elif condition == "vitiligo":
                    response = "Vitiligo is not physically harmful or contagious, but can be psychologically impactful. Depigmented skin has lost natural UV protection, increasing sunburn risk."
                else:
                    response = f"The severity of {condition} varies by individual case. Please consult a healthcare provider for personalized assessment."
            
            else:
                # Default to general description
                template = self._get_response_template("description")
                description = condition_data["description"]
                response = template.format(condition=condition.capitalize(), description=description)
                
                # Add a bit more information for completeness
                symptoms = self._format_list(condition_data["symptoms"], limit=3)
                treatments = self._format_list(condition_data["treatments"], limit=3)
                
                response += f" Common symptoms include {symptoms}. It can be treated with {treatments}."
                
        else:
            # Direct answers to common general questions
            if "what causes skin disease" in prompt_lower or "causes of skin diseases" in prompt_lower:
                response = "Skin diseases can be caused by various factors including genetics, infections (bacterial, viral, or fungal), allergies, immune system disorders, environmental factors like sun exposure, and certain medications. Each specific skin condition has its own set of causes."
            
            elif "when to see doctor" in prompt_lower or "when should i see a doctor" in prompt_lower or "medical attention" in prompt_lower:
                response = "You should see a doctor for skin issues if you notice: persistent rash or irritation that doesn't improve, rapidly spreading rash, fever accompanying a rash, painful skin conditions, signs of infection (swelling, warmth, redness, pus), or changes to existing moles. Any sudden and severe skin reaction should be treated as a medical emergency."
            
            elif "how to maintain" in prompt_lower and "skin health" in prompt_lower or "healthy skin" in prompt_lower:
                response = "To maintain healthy skin: 1) Practice good hygiene with gentle cleansers, 2) Use sunscreen daily with SPF 30+, 3) Stay hydrated, 4) Eat a balanced diet rich in antioxidants, 5) Avoid smoking, 6) Manage stress, 7) Moisturize regularly, 8) Get adequate sleep, 9) Limit alcohol consumption, 10) Exercise regularly to improve circulation."
            
            elif "common skin diseases" in prompt_lower or "types of skin conditions" in prompt_lower:
                response = "Common skin diseases include: acne, eczema, psoriasis, rosacea, vitiligo, hyperpigmentation, hives, shingles, skin cancer, and fungal infections. I can provide more detailed information about acne, hyperpigmentation, nail psoriasis, SJS-TEN, or vitiligo if you're interested."
            
            elif "difference between rash and" in prompt_lower or "what is a rash" in prompt_lower:
                response = "A rash is a general term for any inflammation or discoloration that changes the skin's appearance. Rashes can be symptoms of many different skin conditions, infections, allergies, or systemic diseases. They may be localized or widespread, itchy or painful, flat or raised. Specific conditions like eczema, psoriasis, or contact dermatitis are types of rashes with distinct characteristics."
            
            else:
                # If no specific condition or common question is identified
                template = self._get_response_template("not_found")
                response = template
        
        # Update conversation memory for context awareness
        self.conversation_memory.append((prompt, response))
        if len(self.conversation_memory) > self.memory_size:
            self.conversation_memory.pop(0)  # Remove oldest exchange
            
        # Ensure we don't exceed max tokens
        words = response.split()
        if len(words) > max_tokens:
            response = " ".join(words[:max_tokens]) + "..."
            
        # Calculate and print processing time
        processing_time = time.time() - start_time
        print(f"Response generated in {processing_time:.2f} seconds")
            
        return response
    
    def generate_stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Stream the generated text token by token.
        
        Args:
            prompt: Input text to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (higher = more random)
            
        Yields:
            Generated text tokens
        """
        full_response = self.generate(prompt, max_tokens, temperature)
        words = full_response.split()
        
        for i, word in enumerate(words):
            # Add space before word except for first word
            if i > 0:
                yield " " + word
            else:
                yield word
            
            # Add variation in thinking time for more natural delivery
            time.sleep(0.05 + np.random.random() * 0.1)
    
    def unload(self):
        """Unload the model from memory"""
        if self.model_loaded:
            print("Unloading SkinLLM model...")
            # Clean up resources
            self.knowledge_graph = None
            self.token_bank = None
            self.model_loaded = False
            print("SkinLLM model unloaded successfully!")

class LLMManager:
    """
    Manager class for handling the SkinLLM model lifecycle and inference.
    Manages model loading, unloading, and provides a thread-safe interface
    for generating responses from the model.
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
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
        # Statistics for monitoring
        self.total_requests = 0
        self.avg_response_time = 0
        self.last_response_time = 0
    
    def load_model(self):
        """Load the model if not already loaded"""
        if not self.model_loaded:
            print("Loading knowledge-based LLM for skin diseases...")
            start_time = time.time()
            
            # Initialize our real-time, non-transformer SkinLLM
            self.model = SkinLLM(self.model_path, self.context_size, self.threads)
            
            load_time = time.time() - start_time
            print(f"Model loaded successfully in {load_time:.2f} seconds")
            self.model_loaded = True
    
    def unload_model(self):
        """Unload the model from memory"""
        if self.model_loaded and self.model:
            self.model.unload()
            self.model = None
            self.model_loaded = False
            print("Model resources released")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        Uses the real-time pattern matching and knowledge graph system.
        
        Args:
            prompt: Input text to generate from
            
        Returns:
            Generated text response
        """
        if not self.model_loaded:
            self.load_model()
        
        if self.model:
            start_time = time.time()
            
            # Generate response using our knowledge-based approach
            response = self.model.generate(prompt)
            
            # Update statistics
            self.total_requests += 1
            self.last_response_time = time.time() - start_time
            self.avg_response_time = ((self.avg_response_time * (self.total_requests - 1)) + 
                                     self.last_response_time) / self.total_requests
            
            # Log performance metrics
            if self.total_requests % 10 == 0:
                print(f"Stats: Avg response time: {self.avg_response_time:.3f}s over {self.total_requests} requests")
            
            return response
        else:
            return "Error: Model not loaded. Please try again."
    
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream the generated response token by token for a real-time
        conversational experience. The tokens are generated using
        pattern matching and knowledge retrieval, not neural networks.
        
        Args:
            prompt: Input text to generate from
            
        Yields:
            Generated text tokens
        """
        if not self.model_loaded:
            self.load_model()
        
        if self.model:
            # Update request counter
            self.total_requests += 1
            
            # Stream tokens from our model
            # This provides a realistic typing effect without using transformers
            yield from self.model.generate_stream(prompt)
        else:
            yield "Error: Model not loaded. Please try again."
    
    def start_worker(self):
        """Start the worker thread for processing messages asynchronously"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            print("Background processing worker started")
    
    def stop_worker(self):
        """Stop the worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None
            print("Background processing worker stopped")
    
    def _worker_loop(self):
        """Worker thread function for processing messages in the background"""
        while self.running:
            try:
                # Get message from queue with timeout
                message = self.message_queue.get(timeout=0.1)
                
                # Generate response using our knowledge-based approach
                response = self.generate_response(message)
                
                # Put response in response queue
                self.response_queue.put(response)
                
                # Mark task as done
                self.message_queue.task_done()
            except queue.Empty:
                # Queue is empty, just continue
                pass
            except Exception as e:
                print(f"Error in worker thread: {e}")
                # Put error message in response queue
                self.response_queue.put(f"Sorry, I encountered an error: {str(e)}")
    
    def submit_message(self, message: str):
        """
        Submit a message for asynchronous processing.
        
        Args:
            message: The message to process
        """
        self.message_queue.put(message)
        print(f"Message queued for processing: '{message[:30]}...' (len: {len(message)})")
    
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
            return None
    
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
            "queue_size": self.message_queue.qsize()
        }
    
    def __del__(self):
        """Cleanup when the object is deleted"""
        print("LLMManager cleanup")
        self.stop_worker()
        self.unload_model()
