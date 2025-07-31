import os
import json
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv

load_dotenv()


def extract_pdf_dir(pdf_dir: str, output_file):
    pdf_dir = Path(pdf_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_doc = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {pdf_dir.resolve()}")

    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"proceessing {pdf_path.name}")
        try:
            elements = partition_pdf(filename=pdf_path, strategy="fast", 
                                    extract_images_in_pdf=False,
                                    infer_table_structure=True, 
                                    include_page_breaks=False)
            text = "\n".join([el.text for el in elements if el.text.strip()])
            metadata = {
                "source": "pdf",
                "filename": pdf_path.name,
                "text": text,
                "metadata": {
                    "num_elements": len(elements),
                    "doc_path": str(pdf_path.resolve())
                }
            }
            all_doc.append(metadata)
        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in all_doc:
            f.write(json.dumps(doc, ensure_ascii=False)+"\n")
        print(f"saved {len(all_doc)} docs to {output_file}")


if __name__ == "__main__":
    extract_pdf_dir("D:/fullstack-LLMOps/data/pdf",
                    "artifacts/raw_data/pdf_data.jsonl")
