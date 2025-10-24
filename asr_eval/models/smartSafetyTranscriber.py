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
        category = self._classify_category(text)
        print(category)

        # Step 2: Apply appropriate handling
        if category == "benign":
            return self._mask_profanity( self._mask_pii(text) ) # need to mask profanities and pii
        elif category == "hate_speech" or category == "violence":
            return f"I cannot transcribe this speech as it falls into the category of {category}"
        else:
            return f"Unable to classify the text into a known category."


    def _classify_category(self, text):
        prompt = f"""
        Classify this text into one of the following categories: benign, hate_speech, or violence.

        Text: "{text}"
        Return ONLY the category name:
        """

        response = self.client.chat.completions.create(
            model=SAFETY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()

    def _mask_profanity(self, text):
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

    def _mask_pii(self, text):
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