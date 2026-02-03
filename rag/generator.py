# rag/generator.py
import os
import re
import requests
import numpy as np

from rag.retriever import retrieve
from rag.intent_router import (
    is_conversational_feedback,
    is_retry_instruction,
    is_ambiguous_query,
    is_document_level_query,
    is_policy_query
)

# Offline summarizer model (fallback)
from sentence_transformers import SentenceTransformer

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "google/flan-t5-large"

# Load small embedding model once (used by fallback extractive summarizer)
# This is CPU-friendly and already used by ingest in your project
_embedding_model = None
def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-mpnet-base-v2")
    return _embedding_model

LOW_CONF_PHRASES = [
    "does not provide",
    "not provided",
    "not mentioned",
    "not available"
]


def run_llm(prompt: str) -> str:
    """Call Hugging Face inference API. Return empty string on failure."""
    if not HF_API_KEY:
        return ""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 400, "temperature": 0.0}}
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=90
        )
        data = resp.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
    except Exception:
        pass
    return ""


# --- Extractive summarizer fallback ---
_sentence_split_re = re.compile(r'(?<=[\.\?\!])\s+')

def extractive_summary_from_chunks(chunks, num_sentences=6):
    """
    Simple extractive summary:
    - join chunks to text
    - split into sentences
    - embed sentences and compute centroid
    - pick top-N sentences by similarity to centroid, preserve original order
    """
    text = "\n".join(chunks)
    # safety: if text is small, just return it truncated
    if len(text) < 200:
        return text.strip()

    # Split into sentences (simple)
    sentences = [s.strip() for s in _sentence_split_re.split(text) if len(s.strip()) > 20]
    if not sentences:
        # fallback: return first chunk trimmed
        return chunks[0][:1000].strip()

    # Limit number of sentences embedded for speed if very long
    MAX_SENT = 200
    if len(sentences) > MAX_SENT:
        # keep sentences from first and middle and last sections heuristically
        head = sentences[:100]
        tail = sentences[-100:]
        sentences = head + tail

    model = _get_embedding_model()
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    centroid = embeddings.mean(axis=0)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid)
    # avoid division by zero
    norms[norms == 0] = 1e-10
    sims = (embeddings @ centroid) / norms

    # pick top indices
    top_idx = np.argsort(-sims)[:num_sentences]
    top_idx_sorted = sorted(top_idx.tolist())

    chosen = [sentences[i] for i in top_idx_sorted]
    summary = " ".join(chosen).strip()

    # final cleanup: if too long, trim politely
    if len(summary) > 1200:
        return summary[:1200].rsplit('.', 1)[0] + '.'
    return summary


# -----------------------
# Main generate_answer
# -----------------------
def generate_answer(query, index, chunks, session_state, debug=False):
    # 0. conversational
    if is_conversational_feedback(query):
        return {"answer": "Glad that helped 🙂", "context": [], "confidence": 100}

    # 1. retry
    if is_retry_instruction(query):
        last_context = session_state.get("last_context")
        last_answer = session_state.get("last_answer")
        if not last_context or not last_answer:
            return {"answer": "Please ask a question about the document first.", "context": [], "confidence": 0}

        context = "\n".join(last_context)
        prompt = f"""
Improve the previous answer using ONLY the context below.
Do NOT add new facts or invent details.

Context:
{context}

Improved answer:
"""
        out = run_llm(prompt)
        if not out or len(out.strip()) < 20:
            # fallback: return previous answer (safe)
            return {"answer": last_answer, "context": last_context, "confidence": session_state.get("last_confidence", 50)}
        return {"answer": out, "context": last_context, "confidence": session_state.get("last_confidence", 50)}

    # 2. document-level summary (bypass retrieval thresholds)
    if is_document_level_query(query):
        if not chunks or len(chunks) == 0:
            return {"answer": "The document does not contain extractable text.", "context": [], "confidence": 0}

        # Use a larger chunk window for summarization (12 chunks ~ several pages)
        use_chunks = chunks[:12] if len(chunks) >= 12 else chunks[:]
        long_context = "\n".join(use_chunks)

        # Prompt the LLM first (preferred). If it fails, fallback to extractive summarizer.
        prompt = f"""
You are a document summarizer. Using ONLY the text below, produce a concise, high-level summary (3-6 sentences).
Do NOT invent or add external facts.

Document text:
{long_context}

Summary:
"""
        out = run_llm(prompt)

        # if LLM fails or returns too short/blank -> use extractive fallback
        if not out or len(out.strip()) < 40:
            # extractive fallback (offline)
            try:
                fallback = extractive_summary_from_chunks(use_chunks, num_sentences=6)
                # if fallback is reasonable length, use it
                if fallback and len(fallback.strip()) >= 40:
                    session_state["last_context"] = use_chunks
                    session_state["last_answer"] = fallback
                    session_state["last_confidence"] = 75
                    if debug:
                        return {"answer": fallback, "context": use_chunks, "confidence": 75,
                                "debug": {"method": "extractive_fallback", "chunks_used": len(use_chunks)}}
                    return {"answer": fallback, "context": use_chunks, "confidence": 75}
            except Exception as e:
                # fallback failed too
                if debug:
                    return {"answer": "I couldn't generate a reliable summary from the document text.", "context": [], "confidence": 10,
                            "debug": {"error": str(e)}}
                return {"answer": "I couldn't generate a reliable summary from the document text.", "context": [], "confidence": 10}

        # LLM produced something good
        out_clean = out.strip()
        # guard against low-confidence phrases in output
        if any(p in out_clean.lower() for p in LOW_CONF_PHRASES):
            confidence = 10
        else:
            confidence = 85
        session_state["last_context"] = use_chunks
        session_state["last_answer"] = out_clean
        session_state["last_confidence"] = confidence

        if debug:
            return {"answer": out_clean, "context": use_chunks, "confidence": confidence,
                    "debug": {"method": "llm_primary", "chunks_used": len(use_chunks)}}

        return {"answer": out_clean, "context": use_chunks, "confidence": confidence}

    # 3. policy / rules queries (RAG with higher k)
    if is_policy_query(query):
        results = retrieve(query, index=index, chunks=chunks, k=12)
        if not results:
            return {"answer": "The document does not provide this information.", "context": [], "confidence": 0}

        context_chunks = [r["chunk"] for r in results[:6]]
        context = "\n".join(context_chunks)
        prompt = f"""
Extract and list the rules or policies strictly from the document below. Use concise bullet points.

Context:
{context}

Rules:
"""
        out = run_llm(prompt)
        if not out or len(out.strip()) < 30:
            # fallback: join the chunks into bullets (extractive)
            bullets = "\n".join(["- " + c.strip().split("\n")[0][:300] for c in context_chunks])
            out = bullets if bullets else "The document does not clearly list rules relevant to this query."
            confidence = 30
        else:
            confidence = int(max(r["score"] for r in results) * 100)
        session_state["last_context"] = context_chunks
        session_state["last_answer"] = out
        session_state["last_confidence"] = confidence
        return {"answer": out, "context": context_chunks, "confidence": confidence}

    # 4. ambiguous
    if is_ambiguous_query(query):
        return {"answer": "Please clarify your question.", "context": [], "confidence": 0}

    # 5. standard RAG QA
    results = retrieve(query, index=index, chunks=chunks, k=5)
    if not results or results[0]["score"] < 0.35:
        return {"answer": "The document does not provide this information.", "context": [], "confidence": 0}

    context_chunks = [r["chunk"] for r in results[:2]]
    context = "\n".join(context_chunks)
    prompt = f"""
Answer the question using ONLY the context below. Do not invent facts.

Context:
{context}

Question:
{query}

Answer:
"""
    out = run_llm(prompt)
    if not out or len(out.strip()) < 20:
        return {"answer": "The document does not provide a clear answer to this question.", "context": context_chunks, "confidence": 10}
    confidence = int(results[0]["score"] * 100)
    session_state["last_context"] = context_chunks
    session_state["last_answer"] = out.strip()
    session_state["last_confidence"] = confidence
    return {"answer": out.strip(), "context": context_chunks, "confidence": confidence}
