from pydantic import BaseModel


class Query(BaseModel):
    user_id: str
    question: str
    name: str
    location: str
