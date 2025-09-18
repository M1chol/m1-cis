from __future__ import annotations

import json
from typing import Any, Dict, List

from google import genai
from google.genai import types


class GeminiHandler:
    """
    Python equivalent of the TypeScript geminiHandler using the modern google-genai SDK.

    - Uses model: gemini-2.5-flash-lite (stable and GA as of mid-2025).
    - Supports structured JSON output via response_mime_type and response_schema.
    """

    def __init__(self, google_api_key: str) -> None:
        if not google_api_key:
            raise ValueError("Missing GEMINI_API_KEY. Please provide a valid key.")
        self.client = genai.Client(api_key=google_api_key)
        self.model = "gemini-2.5-flash-lite"

    def get_image_search_pairs(self, context: str) -> List[Dict[str, str]]:
        """
        Ask Gemini to generate 3 (imageDescription, searchQuerry) pairs as JSON.
        Returns a list of dicts with keys: imageDescription, searchQuerry.
        """
        prompt = (
            "Create short search queries that will yield good stock images for the "
            "given context. Images should not be politically biased, be in any way "
            "offensive. If there is a particular persona included, make sure that the search query is simple."
            "Propose 3 image-search pairs.\n"
            f"context: {context}"
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "ImageSearchPair": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "imageDescription": {"type": "string"},
                                    "searchQuerry": {"type": "string"},
                                },
                                "propertyOrdering": [
                                    "imageDescription",
                                    "searchQuerry",
                                ],
                                "required": ["imageDescription", "searchQuerry"],
                            },
                        }
                    },
                    "propertyOrdering": ["ImageSearchPair"],
                    "required": ["ImageSearchPair"],
                },
            ),
        )

        if response is None:
            raise RuntimeError("Unknown generation fail, response undefined")

        json_text = response.text
        if not json_text:
            raise RuntimeError("Response did not contain JSON text")

        try:
            parsed: Any = json.loads(json_text)
        except Exception as e:
            raise RuntimeError(f"Failed to parse AI JSON response: {e}") from e

        if isinstance(parsed, dict) and "ImageSearchPair" in parsed:
            pairs = parsed["ImageSearchPair"]
            if isinstance(pairs, list):
                # Ensure each item is a dict with required keys
                filtered: List[Dict[str, str]] = []
                for item in pairs:
                    if (
                        isinstance(item, dict)
                        and "imageDescription" in item
                        and "searchQuerry" in item
                    ):
                        filtered.append(
                            {
                                "imageDescription": str(item["imageDescription"]),
                                "searchQuerry": str(item["searchQuerry"]),
                            }
                        )
                return filtered

        raise RuntimeError("Invalid response format. Expected ImageSearchPair[]")

    def _test_ai(self) -> str:
        """
        Simple generation test.
        """
        response = self.client.models.generate_content(
            model=self.model, contents="Explain how AI works in a few words"
        )
        return response.text or ""