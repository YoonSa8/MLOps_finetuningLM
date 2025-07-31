import os
import json
import hashlib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import sent_tokenize
import re
from transformers import AutoTokenizer


CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def deduplicats(docs):
    seen = set()
    unique_doc = []
    for doc in docs:
        h = hash_text(doc["text"])
        if h not in seen:
            seen.add(h)
            unique_doc.append(doc)
    return unique_doc


def split_into_chunks(text, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = tokenizer.tokenize(text)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_tokens]
        decoded = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(decoded)
        i += max_tokens - overlap
    return chunks


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def preprocess(cleaned_data, chunked_data):
    docs = load_json(cleaned_data)
    for d in docs:
        d["text"] = clean_text(d.get("text", ""))
    docs = deduplicats(docs)
    save_json(docs, chunked_data)

    chunked = []
    for d in tqdm(docs):
        for idx, chunk in enumerate(split_into_chunks(d['text'])):
            chunked.append({
                "text": chunk,
                "source": d.get("source", ""),
                "chunk_id": f"{hash_text(d['text'])}_{idx}",
                "metadata": {
                    "domain": d.get("metadata", {}).get("domain", ""),
                    "keywords": d.get("metadata", {}).get("domain_tags", []),
                }

            })
    save_json(chunked, chunked_data)

    print(f"preprocessing is complete to {len(chunked)} and been saved")


if __name__ == "__main__":
    cleaned_data = "data/clean/cleaned.jsonl"
    chunked_data = "data/clean/cleaned_data.jsonl"

    preprocess(cleaned_data, chunked_data)
