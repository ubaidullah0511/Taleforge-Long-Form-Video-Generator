from typing import Literal, Protocol

from pydantic import BaseModel


class StockHit(BaseModel):
    source: str
    source_id: str
    download_url: str
    width: int = 0
    height: int = 0
    duration: float = 0
    media_type: Literal["video", "image"] = "video"
    text: str = ""


class StockProvider(Protocol):
    name: str

    async def search(self, query: str, per_page: int = 5) -> list[StockHit]: ...
