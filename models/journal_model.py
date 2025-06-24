from pydantic import BaseModel


class Journal(BaseModel):
    user_id: str
    data: str


class JournalEdit(BaseModel):
    user_id: str
    point_id: str
    new_text: str
