🏦 Banking RAG Platform
Multi-Document Generative AI Assistant for Retail Banking Products

Built with FastAPI • OpenAI • FAISS • Python • RAG • Guardrails • Observability

📌 Overview

This project is an end-to-end Generative AI platform that answers questions about retail banking products using official PDF documents:

Credit Cards

Deposit Accounts

Mortgage Loans

The system implements a production-style RAG (Retrieval-Augmented Generation) pipeline with:

Multi-document ingestion

PDF chunking

Embeddings (OpenAI)

FAISS vector indexing

Guardrails (out-of-scope, unsafe, personal advice filtering)

Confidence thresholding (“I don't know” mode)

Monitoring & audit logging

FastAPI backend endpoints (/ask, /rag-ask)

Designed following banking-grade requirements for explainability, compliance, and traceability.

🧠 Features
✅ 1. Multi-document RAG

Ingests multiple banking PDFs

Splits them into semantic text chunks

Generates embeddings using OpenAI

Stores everything in a FAISS vector index

Answers questions grounded ONLY in official documents

✅ 2. Guardrails

The system blocks:

❌ Out-of-scope queries (crypto, jokes, politics, etc.)

❌ Unsafe content (PIN, passwords, hacking instructions)

❌ Personal/investment advice

✔ Returns compliant fallback responses

✅ 3. Confidence Thresholding

Uses FAISS embedding distances

If the match is weak → returns safe fallback

Prevents hallucinations & ensures answer reliability

✅ 4. Explainability (Sources)

For each answer, the API returns:

Document ID

PDF filename

Page number

Supporting text snippet

✅ 5. Monitoring & Audit Logging

Every query is logged with:

Timestamp

Query

FAISS distances

Retrieved documents/pages

Final LLM answer

Stored in:

logs/rag_logs.txt

✅ 6. FastAPI Backend

Endpoints:

POST /ask       → direct LLM calls  
POST /rag-ask   → banking RAG answers based on PDFs  


Open API Docs:
👉 http://127.0.0.1:8000/docs

📁 Project Structure
genai-banking-rag/
│
├── app.py                     # FastAPI application
├── llm_client.py              # Direct LLM client
├── rag_client.py              # RAG logic (retrieval + LLM)
├── guardrails.py              # Safety rules & filters
├── build_index.py             # Chunking PDFs, embeddings, FAISS index builder
├── requirements.txt
│
├── data/
│   └── credit_cards.pdf
│   └── deposits.pdf
│   └── mortgage_loans.pdf
│
├── logs/
│   └── rag_logs.txt           # Audit log history
│
├── faiss_index_banking.bin    # Vector index
└── banking_chunks.pkl         # Chunk metadata

🏗️ Installation & Setup
git clone https://github.com/<your-username>/genai-banking-rag.git
cd genai-banking-rag

python -m venv venv
source venv/Scripts/activate      # Windows
pip install -r requirements.txt


Create a .env file:

OPENAI_API_KEY=your_key_here

🏦 Building the RAG Index

Place your PDFs inside:

data/


Then run:

python build_index.py


This generates:

faiss_index_banking.bin

banking_chunks.pkl

🚀 Running the API
uvicorn app:app --reload


Visit:

👉 http://127.0.0.1:8000/docs

🧪 Example Query

Request:

{
  "query": "Τι ισχύει για το PIN της πιστωτικής κάρτας;"
}


Response:

{
  "answer": "Το PIN είναι ο προσωπικός κωδικός...",
  "sources": [
    {
      "doc_id": "credit_cards",
      "source_file": "credit_cards.pdf",
      "page": 2,
      "snippet": "Το PIN είναι ο προσωπικός κωδικός..."
    }
  ]
}

🔒 Guardrails in Action
❌ Out-of-scope

Input: “Πες μου ένα ανέκδοτο.”
Output:

Το σύστημα απαντά μόνο σε ερωτήσεις σχετικές με τραπεζικά προϊόντα...

❌ Unsafe

Input: “Πώς να βρω το PIN ενός άλλου;”
→ blocked

❌ Personal advice

Input: “Να πάρω στεγαστικό τώρα;”
→ blocked

📊 Monitoring Example

logs/rag_logs.txt

[2025-01-05 11:20:21] NEW QUERY: Τι είναι το PIN;
[2025-01-05 11:20:21] FAISS distances: [0.12, 0.34, 0.89, 1.03, 1.21]
[2025-01-05 11:20:21] [Rank 1] Doc: credit_cards | Page 2 | Distance: 0.12
[2025-01-05 11:20:21] FINAL_ANSWER: Το PIN είναι...
---- END OF QUERY ----