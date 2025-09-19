from .main import ContextSearch
from PIL.Image import Image
from .types import ImageSearchPair, ImageResult, ImageSearchResult
from typing import List

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