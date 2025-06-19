from typing import Optional
from pydantic import BaseModel


class DailyInsight(BaseModel):
    user_id: str
    is_Subscribed: Optional[bool]
