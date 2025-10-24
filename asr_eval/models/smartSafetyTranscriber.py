SAFETY_MODEL = "llama-3.1-8b-instant"


class SmartSafetyTranscriber:
    def __init__(self, groq_api_key):
        # Delay importing groq until initialization so missing optional deps don't
        # fail at module import time. If groq is unavailable, provide a clear
        # runtime error when attempting to use the transcriber.
        try:
            import importlib
            groq = importlib.import_module("groq")
        except Exception as e:
            raise ImportError(
                "groq package is required for SmartSafetyTranscriber (install via pip)."
            ) from e

        self.client = groq.Groq(api_key=groq_api_key)
    
    def process_text(self, text):
        # Step 1: Classify category
        category = self.classify_category(text)
        
        # Step 2: Apply appropriate handling
        if category == "benign":
            return text
        elif category == "profanity":
            return self.mask_profanity(text)
        elif category == "pii":
            return self.mask_pii(text)  
        else:  # hate_speech or violence
            return f"I cannot transcribe this speech as it falls into this category: {category}"
    
    def classify_category(self, text):
        prompt = f"""
        Classify this text into EXACTLY ONE category: benign, profanity, hate_speech, violence, pii
        
        Text: "{text}"
        Return ONLY the category name:
        """
        
        response = self.client.chat.completions.create(
            model=SAFETY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    
    def mask_profanity(self, text):
        prompt = f"""
        Replace ALL profanity and vulgar words in this text with '[CENSOREDTEXT]'. Keep everything else exactly the same.
        
        Text: "{text}"
        Return only the masked text:
        """
        
        response = self.client.chat.completions.create(
            model=SAFETY_MODEL, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    
    def mask_pii(self, text):
        prompt = f"""
        Mask all personal identifiable information (PII) in this text. Replace with [REDACTED].
        PII includes: phone numbers, NRIC, addresses, email, credit card, personal names, or any other personal idenfitifiable information.
        
        Text: "{text}"
        Return only the masked text:
        """
        
        response = self.client.chat.completions.create(
            model=SAFETY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()