import faiss
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def is_ambiguous_query(query: str) -> bool:
    q = query.lower().strip()
    return len(q.split()) <= 2


def is_document_level_query(query: str) -> bool:
    q = query.lower()
    intents = [
        "summarize", "summary", "overview",
        "what is this document about", "key points"
    ]
    return any(i in q for i in intents)


@lru_cache(maxsize=128)
def embed_query(query: str):
    vec = model.encode([query])
    vec = np.array(vec).astype("float32")
    faiss.normalize_L2(vec)
    return vec


def retrieve(query, index, chunks, k=5):
    query_vec = embed_query(query)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),  # cosine similarity
            "chunk": chunks[idx]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
