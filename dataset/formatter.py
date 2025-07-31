import json
input_path = "./data/qa/qa_together.jsonl"
output_path = "./data/qa/qa_formatted.jsonl"


def format_for_sft(input_path: str, output_path: str):
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            item = json.loads(line)
            formatted = {
                "instruction": item["question"],
                "input": item["context"],
                "output": item["answer"]
            }
            f_out.write(json.dumps(formatted) + "\n")


if __name__ == "__main__":
    format_for_sft(input_path, output_path)
