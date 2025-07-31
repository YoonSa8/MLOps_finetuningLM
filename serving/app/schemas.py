from pydantic import BaseModel


class QueryRequest(BaseModel):
    prompt: str
    max_new_token: int = 200
    temperature: float = 0.7


class QueryResponse(BaseModel):
    output: str
