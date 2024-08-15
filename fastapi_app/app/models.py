from pydantic import BaseModel
from typing import List, Optional

class GameRequest(BaseModel):
    appid: int
    tags: Optional[List[str]]
