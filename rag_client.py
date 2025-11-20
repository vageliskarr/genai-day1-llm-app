import os
import pickle
import logging
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime


# ---------- Logging Setup ----------
import logging

# ---------- Logging Setup ----------
logger = logging.getLogger("rag_client")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("\n[%(levelname)s] %(message)s\n")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Πολύ σημαντικό: να μην το στέλνει και στον uvicorn logger
logger.propagate = False

# ---------- Paths ----------

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_index_banking.bin"
CHUNKS_PATH = BASE_DIR / "banking_chunks.pkl"

# ---------- OpenAI client ----------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

client = OpenAI(api_key=api_key)

# ---------- Lazy-loaded FAISS index + chunks ----------

index = None
chunks_meta = None


def load_index_if_needed():
    global index, chunks_meta

    if index is not None and chunks_meta is not None:
        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            f"Run 'python build_index.py' locally to generate it."
        )
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. "
            f"Run 'python build_index.py' locally to generate it."
        )

    logger.info("Loading FAISS index and banking document chunks...")
    loaded_index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "rb") as f:
        loaded_chunks = pickle.load(f)

    index = loaded_index
    chunks_meta = loaded_chunks

    logger.info(f"Loaded {len(chunks_meta)} chunks across all documents.")


def embed_query(query: str) -> np.ndarray:
    """
    Φτιάχνει embedding για το query του χρήστη.
    """
    logger.info(f"Embedding query: {query}")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    emb = np.array(response.data[0].embedding, dtype="float32")
    return emb.reshape(1, -1)


def retrieve_context(query: str, k: int = 5):
    """
    Κάνει similarity search στο FAISS index και επιστρέφει
    τα top-k πιο σχετικά chunks με metadata και distances.
    """
    logger.info(f"Performing FAISS similarity search (top-k={k})...")

    query_emb = embed_query(query)
    distances, indices = index.search(query_emb, k)

    retrieved = []
    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks_meta):
            meta = chunks_meta[idx]
            logger.info(
                f"[Rank {rank + 1}] Doc: {meta['doc_id']} | File: {meta['source_file']} | "
                f"Page: {meta['page']} | Distance: {distances[0][rank]:.4f}\n"
                f"--- Chunk Content Start ---\n"
                f"{meta['text'][:300]}...\n"
                f"--- Chunk Content End -----"
            )
            retrieved.append(meta)

    # ΕΠΙΣΤΡΕΦΟΥΜΕ ΚΑΙ ΤΙΣ ΑΠΟΣΤΑΣΕΙΣ
    return retrieved, distances[0]


CONFIDENCE_THRESHOLD = 1.2  # μπορείς να το ρυθμίσεις αργότερα


def answer_with_rag(query: str, k: int = 5):
    """
    Multi-document RAG με confidence threshold:
    - Αν οι αποστάσεις είναι μεγάλες, δεν απαντά.
    """
    logger.info(f"\n========== RAG QUERY ==========\nUser Query: {query}\n")

    retrieved, distances = retrieve_context(query, k=k)

    # 1) Αν δεν βρέθηκε τίποτα καθόλου (πολύ σπάνιο αλλά το καλύπτουμε)
    if not retrieved:
        logger.info("No chunks retrieved from FAISS. Returning fallback answer.")
        fallback_answer = (
            "Δεν μπορώ να απαντήσω αξιόπιστα με βάση τα διαθέσιμα έγγραφα."
        )
        return fallback_answer, []

    # 2) Υπολογίζουμε την μικρότερη απόσταση (όσο πιο μικρή τόσο καλύτερο match)
    min_distance = float(min(distances))
    logger.info(f"FAISS distances: {distances.tolist()}")


    logger.info(f"Minimum FAISS distance: {min_distance:.4f}")

    # 3) Αν η καλύτερη απόσταση είναι ΠΟΛΥ μεγάλη, δεν εμπιστευόμαστε το retrieval
    if min_distance > CONFIDENCE_THRESHOLD:
        logger.info(
            f"Min distance {min_distance:.4f} above threshold "
            f"{CONFIDENCE_THRESHOLD:.4f}. Returning 'I don't know' answer."
        )
        low_conf_answer = (
            "Δεν μπορώ να απαντήσω με σιγουριά στην ερώτηση, "
            "με βάση τα έγγραφα που έχουν ενσωματωθεί στο σύστημα."
        )
        return low_conf_answer, []

    # 4) Αν είμαστε ΟΚ με την απόσταση → συνεχίζουμε κανονικό RAG

    # Φτιάχνουμε context όπου φαίνεται και από ποιο προϊόν/σελίδα είναι κάθε κομμάτι
    context_parts = []
    for item in retrieved:
        header = f"[{item['doc_id']} | page {item['page']}]"
        context_parts.append(f"{header}\n{item['text']}")

    context = "\n\n---\n\n".join(context_parts)

    logger.info(
        f"\n========== RAG CONTEXT SENT TO LLM ==========\n"
        f"{context[:1500]}...\n"
        f"=============================================="
    )

    system_prompt = (
        "You are an assistant for answering questions about retail banking products: "
        "credit cards, deposit accounts, and mortgage loans. "
        "Use ONLY the provided context from the bank's official PDF documents. "
        "If the answer is not clearly contained in the context, say that you cannot answer "
        "based on the available information."
    )

    user_message = (
        f"Context from the bank's product documents:\n\n{context}\n\n"
        f"User question: {query}\n\n"
        "Answer in Greek, clearly and concisely. If useful, mention whether the answer "
        "comes from credit cards, deposits or mortgage loans."
    )

    logger.info("Calling LLM with retrieved context...")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    answer = response.choices[0].message.content

    logger.info(
        f"\n========== LLM FINAL ANSWER ==========\n{answer}\n====================================="
    )

    # Ετοιμάζουμε sources για το API
    sources = []
    for item in retrieved:
        sources.append(
            {
                "doc_id": item["doc_id"],
                "source_file": item["source_file"],
                "page": item["page"],
                "snippet": item["text"][:200],
            }
        )

    return answer, sources
