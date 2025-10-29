"""LLM-based safety classifier for post-processing ASR transcripts.

This module provides Layer 3 guardrail functionality using Groq's Llama-3.1-8B
for semantic classification and masking of harmful content.
"""

import os

SAFETY_MODEL = "llama-3.1-8b-instant"


class SmartSafetyTranscriber:
    """LLM-based safety classifier using Groq API.

    Provides semantic analysis for content classification and masking beyond
    keyword-based approaches. Requires GROQ_API_KEY environment variable.
    """

    def __init__(self, groq_api_key=None):
        """Initialize the safety transcriber.

        Parameters
        ----------
        groq_api_key : str, optional
            Groq API key. If not provided, reads from GROQ_API_KEY environment variable.

        Raises
        ------
        ImportError
            If groq package is not installed
        ValueError
            If API key is not provided and not found in environment
        """
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

        # Get API key from parameter or environment variable
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable or pass groq_api_key parameter."
            )

        self.client = groq.Groq(api_key=api_key)

    def process_text(self, text):
        """Process text through LLM-based safety classification and masking.

        Parameters
        ----------
        text : str
            Raw ASR transcript to process

        Returns
        -------
        str
            Processed text with harmful content masked or blocked
        """
        # Step 1: Classify category
        category = self._classify_category(text)
        print(f"[SmartSafetyTranscriber] Category: {category}")

        # Step 2: Apply appropriate handling
        if category == "benign":
            # Still mask profanities and PII even in benign content
            return self._mask_profanity(self._mask_pii(text))
        elif category == "hate_speech" or category == "violence":
            return f"I cannot transcribe this speech as it falls into the category of {category}"
        else:
            return f"Unable to classify the text into a known category."

    def _classify_category(self, text):
        """Classify text into safety categories using LLM.

        Parameters
        ----------
        text : str
            Text to classify

        Returns
        -------
        str
            Category name: "benign", "hate_speech", or "violence"
        """
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
        """Mask profanity using LLM-based detection.

        Parameters
        ----------
        text : str
            Text to process

        Returns
        -------
        str
            Text with profanity replaced by [CENSOREDTEXT]
        """
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
        """Mask PII using LLM-based detection.

        Parameters
        ----------
        text : str
            Text to process

        Returns
        -------
        str
            Text with PII replaced by [REDACTED]
        """
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
