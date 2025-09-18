import os
import sys
from typing import List
from PIL import Image
import pytest
from dotenv import load_dotenv

# Make "src" importable without installing as a package
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from m1_cis import ContextSearch, ImageSearchPair, ImageResult, ImageSearchResult

load_dotenv()

def test_environment_variables() -> None:
    assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY is not set"
    assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY is not set"
    assert os.getenv("GOOGLE_CX"), "GOOGLE_CX is not set"


GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY env variable")
if not GOOGLE_CX:
    raise RuntimeError("Missing GOOGLE_CX env variable")
if not GOOGLE_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY env variable")

cs = ContextSearch(GEMINI_KEY, GOOGLE_KEY, GOOGLE_CX)


def test_ai_basic_text() -> None:
    output = cs._test_ai_simple()
    print(f"basic text response: {output}")
    assert isinstance(output, str)


@pytest.mark.timeout(30)
def test_ai_structured_output() -> None:
    output = cs._test_ai_structured()
    assert isinstance(output, list)
    assert len(output) > 0
    first = output[0]
    assert isinstance(first, ImageSearchPair)
    assert isinstance(first.imageDescription, str) and len(first.imageDescription) > 0
    assert isinstance(first.searchQuerry, str) and len(first.searchQuerry) > 0
    print(f"basic structured response: {first.model_dump()}")


def test_google_images() -> None:
    images = cs._get_images("Donald Tusk")
    assert isinstance(images, list)
    assert len(images) > 0
    first = images[0]
    assert isinstance(first, ImageResult)
    assert isinstance(first.url, str) and len(first.url) > 0
    print(f"google image link: {first.url}")

def test_image_load() -> None:
    image = cs._test_load_image()
    assert isinstance(image, Image.Image)

@pytest.mark.timeout(300)
def test_clip_image_score() -> None:
    score1, score2 = cs._test_clip()
    assert score1 < score2

def test_final_pipeline() -> None:
    images = cs.search(
        "Papież Leon XIV o demokracji, wojnie w Ukrainie i potrzebie przebudzenia"
    )
    assert isinstance(images, list)
    assert len(images) > 0
    first = images[0]
    assert isinstance(first, ImageSearchResult)
    assert isinstance(first.url, str) and len(first.url) > 0
    assert isinstance(first.imageDescription, str) and len(first.imageDescription) > 0
    print(f"\nimage link: {first.url}")
    print(f"search query: {first.imageSearchQuery}")
    print(f"description: {first.imageDescription}")