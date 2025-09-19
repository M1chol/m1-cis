import os
import sys
from PIL import Image
import pytest
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from m1_cis.types import ImageSearchPair, ImageResult, ImageSearchResult
from m1_cis.main import ContextSearchTester

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

def testSearchEnvVars() -> None:
    assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY is not set"
    assert os.getenv("GOOGLE_CX"), "GOOGLE_CX is not set"

if not GOOGLE_KEY: 
    raise ValueError("No GOOGLE_KEY provided")
if not GOOGLE_CX: 
    raise ValueError("No GOOGLE_CX provided")

tester = ContextSearchTester(GOOGLE_API_KEY=GOOGLE_KEY, GOOGLE_CX=GOOGLE_CX)


def test_ai_basic_text() -> None:
    output = tester.test_ai_simple()
    print(f"basic text response: {output}")
    assert isinstance(output, str)

@pytest.mark.timeout(30)
def test_ai_structured_output() -> None:
    output = tester.test_ai_structured()
    assert isinstance(output, list)
    assert len(output) > 0
    first = output[0]
    assert isinstance(first, ImageSearchPair)
    assert isinstance(first.imageDescription, str) and len(first.imageDescription) > 0
    assert isinstance(first.searchQuerry, str) and len(first.searchQuerry) > 0
    print(f"basic structured response: {first.model_dump()}")


def test_google_images() -> None:
    images = tester.test_get_images()
    assert isinstance(images, list)
    assert len(images) > 0
    first = images[0]
    assert isinstance(first, ImageResult)
    assert isinstance(first.url, str) and len(first.url) > 0

def test_image_load() -> None:
    image = tester.test_load_image()
    assert isinstance(image, Image.Image)

@pytest.mark.timeout(300)
def test_clip_image_score() -> None:
    score1, score2 = tester.test_clip()
    assert score1 < score2

def test_final_pipeline() -> None:
    images = tester.test_context_search()
    assert isinstance(images, list)
    assert len(images) > 0
    first = images[0]
    assert isinstance(first, ImageSearchResult)
    assert isinstance(first.url, str) and len(first.url) > 0
    assert isinstance(first.imageDescription, str) and len(first.imageDescription) > 0