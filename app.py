from fastapi import FastAPI
from pydantic import BaseModel
from llm_client import ask_llm
from rag_client import answer_with_rag


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


class RAGRequest(BaseModel):
    query: str

class RAGSource(BaseModel):
    doc_id: str
    source_file: str
    page: int
    snippet: str

class RAGResponse(BaseModel):
    answer: str
    sources: list[RAGSource]

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


@app.post("/rag-ask", response_model=RAGResponse)
def rag_ask_endpoint(request: RAGRequest):
    from guardrails import (
    is_out_of_scope,
    is_unsafe,
    is_personal_advice,
    classify_product
   )
    
    query = request.query

    # 1. Out-of-scope
    if is_out_of_scope(query):
        return RAGResponse(
            answer="Το σύστημα απαντά μόνο σε ερωτήσεις για πιστωτικές κάρτες, "
                "λογαριασμούς καταθέσεων και στεγαστικά δάνεια, βασισμένο σε επίσημα έγγραφα.",
            sources=[]
        )

    # 2. Unsafe
    if is_unsafe(query):
        return RAGResponse(
            answer="Δεν μπορώ να απαντήσω σε αιτήματα που περιέχουν ευαίσθητες ή μη ασφαλείς πληροφορίες.",
            sources=[]
        )

    # 3. Personal advice (compliance)
    if is_personal_advice(query):
        return RAGResponse(
            answer="Δεν παρέχω οικονομικές ή προσωπικές συμβουλές. "
                "Μπορώ να εξηγήσω μόνο όσα περιλαμβάνονται στα επίσημα έγγραφα.",
            sources=[]
        )

    # 4. Classifier (optional)
    product = classify_product(query)
    # Μπορείς να το χρησιμοποιήσεις αν θες στο RAG για filtering στο μέλλον

    answer, sources = answer_with_rag(request.query)
    return RAGResponse(answer=answer, sources=sources)

