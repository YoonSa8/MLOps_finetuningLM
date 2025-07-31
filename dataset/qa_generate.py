import json
import os
from pathlib import Path
from tqdm import tqdm
from together import Together
import dotenv

dotenv.load_dotenv(dotenv_path="config/.env")
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
input_path = "./data/clean/chunked.jsonl"
output_path = "./data/qa/qa_together.jsonl"


def generate_qa_from_text(input_path: str, output_path: str, model=model_name, max_ques=3):
    with open(input_path, 'r', encoding='utf-8')as f:
        chunks = [json.loads(line) for line in f]
    with open(output_path, 'a', encoding='utf-8') as out_f:
        for chunk in tqdm(chunks):
            prompt = prompt = f"""
You are a helpful assistant. Read the following document chunk and generate high-quality, clear question-answer pairs.
Each question must be answerable only using the context below.

Context: {chunk['text']}

Provide {max_ques} question-answer pairs in JSON.
Return a JSON array in this format:
[
    {{
        "question": "...",
        "answer": "..."
    }}
]
            """
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024
                )
                qa_list = json.loads(response.choices[0].message.content)
                for qa in qa_list:
                    qa_obj = {
                        "context": chunk["text"],
                        "question": qa['question'],
                        "answer": qa['answer'],
                        "source": chunk['source'],
                        "metadata": chunk['metadata']
                    }
                    out_f.write(json.dumps(qa_obj)+"\n")

            except Exception as e:
                print(f'failed on chunk {chunk.get("chunk_id", "")}: {e}')


if __name__ == "__main__":
    generate_qa_from_text(input_path, output_path)
