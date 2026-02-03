# 📘 Personal RAG Assistant (PDF-based)

An **end-to-end Retrieval-Augmented Generation (RAG)** assistant that allows users to upload PDF documents and ask questions or request summaries based strictly on the document content.

This project focuses on **accuracy, document grounding, and clean retrieval**, avoiding hallucinations by design.

---

## ✨ Features

- 📄 Upload and analyze PDF documents  
- 🔍 Semantic search using **FAISS**  
- 🧠 Context-aware answers using **Sentence Transformers**  
- 📊 Confidence-aware responses (**High / Medium / Low**)  
- 🧼 Robust PDF text normalization (handles broken-line PDFs)  
- 🚫 Hallucination-safe: answers strictly from document context  
- 🖥️ Simple, clean **Streamlit UI**

---

## 🧠 Architecture Overview

### 1️⃣ PDF Ingestion
- Text extracted using `pypdf`
- Normalized to fix broken line breaks and formatting issues

### 2️⃣ Chunking & Embeddings
- Text split into meaningful chunks
- Embeddings generated using `sentence-transformers`

### 3️⃣ Vector Search
- **FAISS** used for fast similarity-based retrieval

### 4️⃣ Answer Generation
- Relevant chunks retrieved from vector store
- Answers generated only from retrieved content
- Confidence estimated based on retrieval strength

---

## 📂 Project Structure

personal-rag-bot/
├── app.py # Streamlit application

├── requirements.txt # Python dependencies

├── .gitignore # Ignored files (venv, cache, binaries)

├── rag/

│ ├── ingest.py # PDF ingestion & normalization

│ ├── retriever.py # FAISS retrieval logic

│ ├── generator.py # Answer generation & confidence logic

│ ├── intent_router.py # Query intent detection

│ └── init.py


---

## ⚙️ Setup Instructions (Local)

### 1️⃣ Clone the Repository
```
git clone https://github.com/KovidhN/personal-rag-bot.git
cd personal-rag-bot
2️⃣ Create and Activate Virtual Environment
python -m venv .venv
Activate (Windows – PowerShell):

.\.venv\Scripts\activate
You should see:

(.venv)
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
streamlit run app.py
The app will open automatically in your browser.

🧪 How to Use
Upload a PDF document from the sidebar
Wait for indexing to complete

Ask questions such as:
summarize
what is this document about
what are the key points
does the document mention X?

View:
Answer
Confidence level
Explanation (optional)

⚠️ Important Notes
This project does NOT use external LLM APIs
Works fully on CPU
No internet required after dependency installation
.venv and generated artifacts are intentionally excluded from GitHub

🛠️ Dependencies
Key libraries used:
streamlit
pypdf
faiss-cpu
sentence-transformers
numpy

All dependencies are listed in requirements.txt.

🎯 Design Philosophy
Accuracy over verbosity
Transparency over hallucination
Production-style safeguards
Readable, maintainable code

🧑‍💻 Author
Kovidh Nougain
B.Tech – AI & Data Science
Focus areas: NLP, RAG systems, ML pipelines

📌 Future Improvements
Multi-document support
Caching embeddings across sessions
GPU acceleration
Advanced summarization modes
Deployment on cloud platforms

⭐ If you found this useful
Feel free to ⭐ star the repository or fork it for experimentation.







