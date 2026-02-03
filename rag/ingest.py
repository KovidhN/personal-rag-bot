import faiss
import tempfile
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import re

def normalize_pdf_text(text: str) -> str:
    """
    Fix broken PDF text:
    - Remove word-level newlines
    - Reconstruct paragraphs
    - Normalize spaces
    """

    if not text:
        return ""

    # Remove hyphenated line breaks: "oppor-\ntunity" → "opportunity"
    text = re.sub(r"-\n", "", text)

    # Replace single newlines between words with space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Replace multiple newlines with paragraph break
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def build_index_from_pdf(uploaded_file):
    """
    Safely extract text from PDF, build embeddings, and create FAISS index.
    Handles broken fonts and malformed PDFs gracefully.
    """

    # -------------------------------
    # 1. Save uploaded PDF temporarily
    # -------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    reader = PdfReader(pdf_path)

    full_text = []

    # -------------------------------
    # 2. SAFE text extraction
    # -------------------------------
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                full_text.append(text)
        except Exception as e:
            # Skip problematic pages instead of crashing
            if __debug__:
             print(f"[WARN] Skipping page {i}: {e}")


    if not full_text:
        raise ValueError("No extractable text found in the PDF.")

    document_text = "\n".join(full_text)

    # -------------------------------
    # 3. Chunking (resume-friendly)
    # -------------------------------
    chunks = []
    chunk_size = 400

    for i in range(0, len(document_text), chunk_size):
        chunk = document_text[i:i + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)

    # -------------------------------
    # 4. Embedding (CPU-optimized)
    # -------------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=False)

    # -------------------------------
    # 5. FAISS index
    # -------------------------------
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks
