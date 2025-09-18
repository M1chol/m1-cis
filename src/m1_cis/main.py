from __future__ import annotations

import os
from typing import List, Optional

import requests
from pydantic import BaseModel

from .clip_handler import CLIPHandler
from .gemini_handler import GeminiHandler


class ImageSearchPair(BaseModel):
    imageDescription: str
    searchQuerry: str


class ImageResult(BaseModel):
    id: str
    url: str
    title: str
    source: str
    sourceName: str
    thumbnail: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class ImageSearchResult(BaseModel):
    imageDescription: str
    imageSearchQuery: str
    url: str


class ContextSearch:
    """
    - Uses google Custom Search (image) for results
    - Uses Gemini 2.5 Flash-Lite to propose search pairs
    - Ranks with CLIP zero-shot image classification
    """

    def __init__(self, google_api_key: str, search_api_key: str, search_cx: str) -> None:
        if not google_api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY. Please provide a valid key."
            )
        if not search_api_key:
            raise ValueError("Missing GOOGLE_API_KEY. Please provide a valid key.")
        if not search_cx:
            raise ValueError("Missing GOOGLE_CX. Please provide a valid key.")

        self.cx = search_cx
        self.search_api_key = search_api_key
        self.ai = GeminiHandler(google_api_key)
        self.clip = CLIPHandler()
        self.clip.init()

    def _test_ai_simple(self) -> str:
        return self.ai._test_ai()

    def _test_ai_structured(self) -> List[ImageSearchPair]:
        raw_pairs = self.ai.get_image_search_pairs(
            "Rheinmetall to Acquire German Naval Shipbuilder NVL"
        )
        return [ImageSearchPair(**p) for p in raw_pairs]

    def _test_clip(self) -> float:
        return self.clip._test_clip()

    def _test_load_image(self):
        link = "https://freetestdata.com/wp-content/uploads/2022/02/Free_Test_Data_117KB_JPG.jpg"
        image = self.clip._load_image(link)
        return image

    def _get_images(self, query: str) -> List[ImageResult]:
        """
        Query Google Custom Search for images.
        """
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "cx": self.cx,
            "searchType": "image",
            "key": self.search_api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        if not resp.ok:
            raise RuntimeError("Google search failed")

        data = resp.json()
        if not data:
            raise RuntimeError("Failed to parse response")

        items = data.get("items", []) or []
        images: List[ImageResult] = []
        for index, item in enumerate(items):
            image_info = (item.get("image") or {}) if isinstance(item, dict) else {}
            images.append(
                ImageResult(
                    id=item.get("cacheId") or str(index),
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    source=image_info.get("contextLink", "") or "",
                    sourceName=image_info.get("contextLink", "") or "",
                    thumbnail=image_info.get("thumbnailLink"),
                    width=image_info.get("width"),
                    height=image_info.get("height"),
                )
            )
        return images

    def search(self, context: str, batch_size: int = 8, limit: int = 3, custom_prompt: str = "") -> List[ImageSearchResult]:
        """
        Full pipeline (embedding-based ranking):
        - Use Gemini to propose (imageDescription, searchQuerry)
        - Fetch images via Google Image Search for each query
        - Rank all returned images by CLIP cosine similarity to imageDescription
        - Return best URL per pair (empty if none valid)
        """
        pairs_raw = self.ai.get_image_search_pairs(context)
        pairs = [ImageSearchPair(**p) for p in pairs_raw]

        results: List[ImageSearchResult] = []

        for pair in pairs:
            images = self._get_images(pair.searchQuerry, limit=limit, custom_prompt=custom_prompt)
            if not images:
                results.append(
                    ImageSearchResult(imageDescription=pair.imageDescription, url="", imageSearchQuery=pair.searchQuerry)
                )
                continue

            # Collect candidate URLs; dedupe to avoid redundant downloads
            seen = set()
            image_urls = []
            for img in images:
                if img.url and img.url not in seen:
                    seen.add(img.url)
                    image_urls.append(img.url)

            if not image_urls:
                results.append(
                    ImageSearchResult(imageDescription=pair.imageDescription, url="", imageSearchQuery=pair.searchQuerry)
                )
                continue

            # Rank all images against the single description (batched, robust to failures)
            ranked = self.clip.rank_images_by_description(
                description=pair.imageDescription,
                image_urls=image_urls,
                batch_size=batch_size,
            )

            # ranked is List[Tuple[url, score]] sorted desc; skip if all failed (-1.0)
            best_url = ""
            if ranked:
                top_url, top_score = ranked[0]
                if top_score > -1.0:
                    best_url = top_url

            results.append(
                ImageSearchResult(
                    imageDescription=pair.imageDescription,
                    url=best_url,
                    imageSearchQuery=pair.searchQuerry,
                )
            )

        return results