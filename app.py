from fastapi import FastAPI
from pydantic import BaseModel
from llm_client import ask_llm

# Create FastAPI application instance
app = FastAPI(
    title="GenAI Day 1 – LLM API",
    description="Simple LLM-powered question answering microservice.",
    version="0.1.0",
)

# ----------- Request & Response Models -----------

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    answer: str

# ----------- Root Endpoint ("/") -----------

@app.get("/")
def root():
    return {
        "message": "GenAI LLM API is running.",
        "tip": "Visit /docs for interactive API documentation."
    }

# ----------- Health Check -----------

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ----------- Main LLM Endpoint -----------

@app.post("/ask", response_model=PromptResponse)
def ask_endpoint(request: PromptRequest):
    answer = ask_llm(request.prompt)
    return PromptResponse(answer=answer)
