from pydantic import BaseModel
from typing import Optional

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