from pydantic import BaseModel
from typing import List, Optional

class GameRequest(BaseModel):
    appid: int # The unique identifier for the game (Steam App ID).
    tags: Optional[List[str]]  # An optional list of tags associated with the game.
