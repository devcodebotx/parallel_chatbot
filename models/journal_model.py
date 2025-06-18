from pydantic import BaseModel


class Journal(BaseModel):
    user_id: str
    data: str
