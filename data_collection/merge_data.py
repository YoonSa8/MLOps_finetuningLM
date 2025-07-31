import json


def merge_files(input_files: list[str], output_file):
    merged = []
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                merged.append(json.loads(line))

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in merged:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"Merged {len(merged)} entries into {output_file}")


if __name__ == "__main__":
    merge_files(input_files=["artifacts/processed/pdf_data_enriched.jsonl", "artifacts/processed/web_data_enriched.jsonl"],
                output_file="data/clean/raw.jsonl")
