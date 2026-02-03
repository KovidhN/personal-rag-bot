import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from preprocessing.clean_text import extract_text, clean_text

DATA_DIR = "data"

model = SentenceTransformer("all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

all_chunks = []

for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)
    text = extract_text(path)
    cleaned = clean_text(text)
    chunks = splitter.split_text(cleaned)
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")

embeddings = model.encode(all_chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "embeddings.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("FAISS index + chunks saved successfully")
