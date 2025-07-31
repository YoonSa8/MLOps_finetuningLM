from fastapi import FastAPI, Header, HTTPException, Depends
from app.schemas import QueryRequest, QueryResponse
from app.model import generate_answer
import os 
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Finetuned QA model")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/generate_text", response_model=QueryResponse, dependencies=[Depends(verify_token)])
def generate_text(request: QueryRequest):
    output = generate_answer(
        request.prompt, request.max_new_token, request.temperature)
    return QueryResponse(output=output)
