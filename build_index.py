import os
import pickle
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# --------- Paths & Documents ---------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DOCS = [
    {"doc_id": "credit_cards", "file": "credit_cards.pdf"},
    {"doc_id": "deposits", "file": "deposits.pdf"},
    {"doc_id": "mortgage_loans", "file": "mortgage_loans.pdf"},
]

INDEX_PATH = BASE_DIR / "faiss_index_banking.bin"
CHUNKS_PATH = BASE_DIR / "banking_chunks.pkl"

# --------- OpenAI client ---------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

client = OpenAI(api_key=api_key)


def load_pdf_pages(pdf_path: Path):
    """
    Διαβάζει τις σελίδες ενός PDF και επιστρέφει λίστα (page_num, text).
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150):
    """
    Κόβει κείμενο σε chunks με overlap.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_chunks_for_all_docs():
    """
    Φτιάχνει λίστα από chunks με metadata:
    [
      {
        "text": "...",
        "doc_id": "credit_cards",
        "source_file": "credit_cards.pdf",
        "page": 3
      },
      ...
    ]
    """
    all_chunks = []

    for doc in DOCS:
        doc_id = doc["doc_id"]
        filename = doc["file"]
        pdf_path = DATA_DIR / filename

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found for {doc_id} at {pdf_path}")

        print(f"📄 Loading PDF for {doc_id}: {filename}")
        pages = load_pdf_pages(pdf_path)

        for page_num, page_text in pages:
            page_chunks = chunk_text(page_text, chunk_size=700, overlap=150)
            for ch in page_chunks:
                all_chunks.append(
                    {
                        "text": ch,
                        "doc_id": doc_id,
                        "source_file": filename,
                        "page": page_num,
                    }
                )

    print(f"Total chunks across all documents: {len(all_chunks)}")
    return all_chunks


def embed_texts(texts):
    """
    Δημιουργεί embeddings για λίστα κειμένων.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    embeddings = [np.array(item.embedding, dtype="float32") for item in response.data]
    return np.vstack(embeddings)


def build_faiss_index(embeddings: np.ndarray):
    """
    Φτιάχνει FAISS index (L2 distance).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main():
    print("📄 Building multi-document RAG index (cards + deposits + mortgage loans)...")

    chunks_with_meta = build_chunks_for_all_docs()

    print("🧠 Creating embeddings for all chunks...")
    texts = [item["text"] for item in chunks_with_meta]
    embeddings = embed_texts(texts)
    print("Embeddings shape:", embeddings.shape)

    print("📦 Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"💾 Saving index to {INDEX_PATH} and chunks metadata to {CHUNKS_PATH}...")
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks_with_meta, f)

    print("✅ Multi-document RAG index built successfully.")


if __name__ == "__main__":
    main()
