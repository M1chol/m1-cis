from __future__ import annotations

from typing import List, cast

from google import genai
from google.genai import types

from .types import ImageSearchPair

class GeminiHandler:
    """
    Python equivalent of the TypeScript geminiHandler using the modern google-genai SDK.

    - Uses model: gemini-2.5-flash-lite (stable and GA as of mid-2025).
    - Supports structured JSON output via response_mime_type and response_schema.
    """

    def __init__(self, google_api_key: str) -> None:
        if not google_api_key:
            raise ValueError("Missing GEMINI_API_KEY. Please provide a valid key.")
        try:
            self.client = genai.Client(api_key=google_api_key)
        except:
            raise ValueError("[m1-cis] GeminiHandler - wrong api key, did you enable Generative Language API?")
        self.model = "gemini-2.5-flash-lite"

    def get_image_search_pairs(self, context: str, limit: int, custom_prompt: str) -> List[ImageSearchPair]:
        """
        Ask Gemini to generate 3 (imageDescription, searchQuerry) pairs as JSON.
        Returns a list of dicts with keys: imageDescription, searchQuerry.
        """
        base = (
            "Create short search queries that will yield stock images for the given context. "
            "If context specifies a name, create one pair containing explicit search for it. "
            )

        prompt = base + custom_prompt + (
            f"Propose {limit if limit > 0 else 1 : } image-search pairs.\n"
            f"context: {context}"
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[ImageSearchPair]
            ),
        )
        parsed: list[ImageSearchPair]
        if response.parsed is not None:
            parsed = cast(list[ImageSearchPair], response.parsed)
        else:
            raise ValueError("Failed to parse AI response")
        
        return parsed