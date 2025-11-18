GenAI Day 1 – LLM Question Answering API

A minimal Generative AI microservice built with FastAPI and the OpenAI API.
It takes a user prompt and returns an LLM-generated answer through a REST API.

🚀 Features

Python-based microservice for LLM question answering

FastAPI backend with automatic Swagger UI (/docs)

Secure API key handling through .env

Modular design (llm_client.py handles all LLM interactions)

Interactive console tester (test_llm.py)

Uvicorn ASGI server

Ready for further extension into RAG, embeddings, agents

🏗 Project Structure
genai-day1-llm-app/
├── app.py             # FastAPI REST API
├── llm_client.py      # OpenAI client wrapper
├── test_llm.py        # Console-based LLM tester
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignored files (venv, env)
└── .env               # Environment variables (NOT in repo)

🔧 Installation

Clone the repository:

git clone https://github.com/<username>/genai-day1-llm-app.git
cd genai-day1-llm-app


Create and activate a virtual environment:

python -m venv venv
source venv/Scripts/activate     # Git Bash / Linux / macOS


Install dependencies:

pip install -r requirements.txt


Create a .env file:

OPENAI_API_KEY=sk-proj-xxxxxxxxxxxx

🖥 Console Testing (manual testing)

Run:

python test_llm.py


Example:

You: Give me 3 GenAI use cases in banking.
AI: ...


Type exit to quit.

🌐 Running the API

Start the server:

uvicorn app:app --reload


Open in browser:

Swagger UI → http://127.0.0.1:8000/docs

Health check → http://127.0.0.1:8000/health

Root message → http://127.0.0.1:8000/

📮 API Example
POST /ask

Request:

{
  "prompt": "Give me 3 GenAI risks in banking."
}


Response:

{
  "answer": "1. ... 2. ... 3. ..."
}

📌 Notes

This project represents the first step in building a full GenAI stack (Day 1).
Next stages include:

RAG (Retrieval-Augmented Generation)

Embeddings

Vector stores

Agentic workflows

Monitoring and observability