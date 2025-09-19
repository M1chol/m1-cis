from __future__ import annotations
from typing import Dict, List, Set
from PIL.Image import Image
import requests
from transformers import Any

from .types import ImageSearchPair, ImageResult, ImageSearchResult
from .clip_handler import CLIPHandler
from .gemini_handler import GeminiHandler


class ContextSearch:
    def __init__(self, *, GOOGLE_API_KEY: str, GOOGLE_CX: str) -> None:
        if not GOOGLE_API_KEY:
            raise ValueError("Missing GOOGLE_API_KEY. Please provide a valid key.")
        if not GOOGLE_CX:
            raise ValueError("Missing GOOGLE_CX. Please provide a valid id.")

        self.cx = GOOGLE_CX
        self.google_api_key = GOOGLE_API_KEY
        self.ai = GeminiHandler(GOOGLE_API_KEY)
        self.clip = CLIPHandler()
        self.clip.init()

    def _get_images(self, query: str) -> List[ImageResult]:
        """
        Query Google Custom Search for images.
        """
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "cx": self.cx,
            "searchType": "image",
            "key": self.google_api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        if not resp.ok:
            raise RuntimeError("Google search failed")

        data = resp.json()
        if not data:
            raise RuntimeError("Failed to parse response")

        items: List[Dict[str, Any]] = data.get("items", []) or []
        images: List[ImageResult] = []
        for item in items:
            image_info: Dict[str, str] = item.get("image", {})
            images.append(
                ImageResult(
                    id=item.get("cacheId", ""),
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    source=image_info.get("contextLink", "") or "",
                    sourceName=image_info.get("contextLink", "") or "",
                    thumbnail=image_info.get("thumbnailLink"),
                    width=int(image_info.get("width", "")),
                    height=int(image_info.get("height", "")),
                )
            )
        return images

    def searchWithQuery(self, query: str) -> List[ImageResult]:
        images = self._get_images(query=query)
        return images

    def searchWithContext(self, context: str, batch_size: int = 8, limit: int = 3, custom_prompt: str = "") -> List[ImageSearchResult]:
        """
        Full pipeline (embedding-based ranking):
        - Use Gemini to propose (imageDescription, searchQuerry)
        - Fetch images via Google Image Search for each query
        - Rank all returned images by CLIP cosine similarity to imageDescription
        - Return best URL per pair (empty if none valid)
        """
        pairs = self.ai.get_image_search_pairs(context, limit=limit, custom_prompt=custom_prompt)

        results: List[ImageSearchResult] = []

        for pair in pairs:
            images = self._get_images(pair.searchQuerry)
            if not images:
                results.append(
                    ImageSearchResult(imageDescription=pair.imageDescription, url="", imageSearchQuery=pair.searchQuerry)
                )
                continue

            # Collect candidate URLs; dedupe to avoid redundant downloads
            seen: Set[str] = set()
            image_urls: List[str] = []
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
    
class ContextSearchTester(ContextSearch):
    def test_ai_simple(self) -> str:
        response = self.ai.client.models.generate_content(
            model=self.ai.model, contents="Explain how AI works in a few words"
        )
        return response.text if response.text else ""
    
    def test_ai_structured(self) -> List[ImageSearchPair]:
        pairs = self.ai.get_image_search_pairs(
            "Rheinmetall to Acquire German Naval Shipbuilder NVL",
            limit=2,
            custom_prompt=""
        )
        return pairs

    def test_clip(self) -> tuple[float, float]:
        return self.clip.test_clip()
    
    def test_get_images(self) -> List[ImageResult]:
        image = self._get_images("Donald tusk")
        return image

    def test_load_image(self) -> Image: 
        link = "https://freetestdata.com/wp-content/uploads/2022/02/Free_Test_Data_117KB_JPG.jpg"
        image = self.clip.load_image(link)
        return image
    
    def test_context_search(self) -> List[ImageSearchResult]:
        images = self.searchWithContext("Papież Leon XIV o demokracji, wojnie w Ukrainie i potrzebie przebudzenia")
        return images