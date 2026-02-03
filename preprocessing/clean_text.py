import spacy
from pypdf import PdfReader
import os

nlp = spacy.load("en_core_web_sm")

def extract_text(path):
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def clean_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)

if __name__ == "__main__":
    print("SCRIPT STARTED")

    data_files = os.listdir("data")
    print("Files in data folder:", data_files)

    if not data_files:
        print("❌ data folder is empty")
        exit()

    data_file = os.path.join("data", data_files[0])
    print("Reading file:", data_file)

    raw_text = extract_text(data_file)
    print("RAW TEXT LENGTH:", len(raw_text))

    cleaned = clean_text(raw_text)

    print("\n----- RAW TEXT -----")
    print(raw_text)

    print("\n----- CLEANED TEXT -----")
    print(cleaned)

