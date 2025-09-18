from __future__ import annotations

from io import BytesIO
from typing import List, Optional, Sequence, Tuple

import requests
from PIL import Image
import torch
from transformers import (
    CLIPModel,
    AutoTokenizer,
    AutoImageProcessor,
)


class CLIPHandler:
    """
    Evaluate how well different images match a given description using CLIP embeddings.

    - Loads CLIPModel + explicit tokenizer + image processor (no CLIPProcessor.__call__).
    - Provides:
        - text_image_similarity: score a single image vs a description
        - rank_images_by_description: rank many images vs one description (batched)
    """

    model: Optional[CLIPModel]
    tokenizer: Optional[object]
    image_processor: Optional[object]

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14") -> None:
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Choose reasonable dtype
        if self.device.type == "cuda":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

    def init(self) -> None:
        """
        Initialize CLIP model, tokenizer, and image processor.
        """
        # Explicit components avoid the CLIPProcessor fast-image-processor bug
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_image(self, url: str) -> Image.Image:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, stream=True, timeout=20, headers=headers)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CLIPHandler not initialized. Call .init() first.")
        enc = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        text_embeds = self.model.get_text_features(**enc)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds  # [1, d]

    @torch.no_grad()
    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if self.model is None or self.image_processor is None:
            raise RuntimeError("CLIPHandler not initialized. Call .init() first.")
        enc = self.image_processor(images=list(images), return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        image_embeds = self.model.get_image_features(**enc)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds  # [N, d]

    def text_image_similarity(self, link: str, text: str) -> float:
        """
        Compute cosine similarity between an image (by URL) and a text description.
        Returns a scalar score, higher is better. Returns -1.0 on load failure.
        """
        if self.model is None:
            raise RuntimeError("CLIPHandler not initialized. Call .init() first.")

        try:
            image = self._load_image(link)
        except Exception as e:
            print(f"[WARN] Failed to load {link}: {e}")
            return -1.0

        with torch.no_grad():
            t = self._encode_text(text)  # [1, d]
            v = self._encode_images([image])  # [1, d]
            sim = torch.matmul(v, t.T)  # [1, 1]
            return float(sim.item())

    def rank_images_by_description(
        self,
        description: str,
        image_urls: Sequence[str],
        batch_size: int = 8,
    ) -> List[Tuple[str, float]]:
        """
        Rank multiple images by similarity to the given description.

        - Downloads images (skips failures)
        - Encodes in batches for GPU/CPU efficiency
        - Returns list of (url, score) sorted desc
        """
        if self.model is None:
            raise RuntimeError("CLIPHandler not initialized. Call .init() first.")

        # Load images with robust handling
        loaded: List[Tuple[int, str, Optional[Image.Image]]] = []
        for i, url in enumerate(image_urls):
            try:
                img = self._load_image(url)
                loaded.append((i, url, img))
            except Exception as e:
                print(f"[WARN] Failed to load {url}: {e}")
                loaded.append((i, url, None))

        with torch.no_grad():
            text_emb = self._encode_text(description)  # [1, d]

        results: List[Tuple[str, float]] = []
        batch: List[Tuple[int, str, Image.Image]] = []
        for i, url, img in loaded:
            if img is None:
                results.append((url, -1.0))
                continue
            batch.append((i, url, img))
            if len(batch) >= batch_size:
                results.extend(self._score_batch(batch, text_emb))
                batch = []
        if batch:
            results.extend(self._score_batch(batch, text_emb))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @torch.no_grad()
    def _score_batch(
        self, batch: List[Tuple[int, str, Image.Image]], text_emb: torch.Tensor
    ) -> List[Tuple[str, float]]:
        images = [img for _, _, img in batch]
        image_embs = self._encode_images(images)  # [B, d]
        sims = torch.matmul(image_embs, text_emb.T).squeeze(-1)  # [B]
        return [(url, float(score.item())) for (_, url, _), score in zip(batch, sims)]

    def _test_clip(self) -> Tuple[float, float]:
        """
        Example test: compare two URLs vs the same description.
        """
        hoodie_desc = "a tiger wearing a hoodie"
        url_hoodie = (
            "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg"
        )
        url_hat = (
            "https://external-preview.redd.it/C73VSiHc_VMC6jgtyHpHNWdpeF6JXq9-"
            "laYNScVdK48.png?width=640&crop=smart&auto=webp&"
            "s=be21d998834673bc059a75fd342e5689c0c6d792"
        )

        ranked = self.rank_images_by_description(hoodie_desc, [url_hoodie, url_hat], batch_size=2)
        scores_by_url = {u: s for u, s in ranked}
        score_hoodie = scores_by_url.get(url_hoodie, -1.0)
        score_hat = scores_by_url.get(url_hat, -1.0)
        return score_hoodie, score_hat