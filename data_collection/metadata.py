import json 
import uuid 
import re 
from pathlib import Path
from datetime import datetime

def extract_tags(text: str, domain_keywords:list[str]) -> list[str]:
    text_lower = text.lower()
    return[kw for kw in domain_keywords if kw in text_lower]

def enrich_metadata(input_path: str, output_path:str, domain_keywords:list[str]):
    input_path= Path(input_path)
    output_path= Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enriched =[]
    with open(input_path,'r',encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc["source_id"]= str(uuid.uuid4())
            base = doc.get("filename") or doc.get("url") or "unkown"
            doc["metadata"]["domain_tags"]=extract_tags(doc["text"], domain_keywords)
            enriched.append(doc)

    with open(output_path, "w", encoding="utf-8")as f:
        for doc in enriched:
            f.write(json.dumps(doc, ensure_ascii=False)+ "\n")
        print(f"enriched {len(enriched)} documents")

if __name__=="__main__":
    keywords = ["ev", "charging", "stations", "infrastructure", "battery", "electrical","energy", "vehicles"]
    enrich_metadata("artifacts/raw_data/pdf_data.jsonl", "artifacts/processed/pdf_data_enriched.jsonl", keywords)
    enrich_metadata("artifacts/raw_data/web_data.jsonl", "artifacts/processed/web_data_enriched.jsonl", keywords)
