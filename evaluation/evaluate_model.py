import json 
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from .helpers  import compute_metrics, timed_generated
from tqdm import tqdm

def load_eval_dataset(path ="/data/evaluation/eval_dataset.jsonl"):
    with open(path, 'r', encoding="utf-8") as f :
        return[json.loads(line) for line in f]
    
def run_eval(model_dir , dataset):
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "auto")
    tokenizer= AutoTokenizer.from_pretrained(model_dir)
    pipe = pipeline("text-generation", model = model, tokenizer= tokenizer)

    preds,refs,times= [],[],[]
    for sample in tqdm(dataset):
        generated, latency = timed_generated(pipe, sample["instruction"])
        preds.append(generated.split(sample["instruction"])[-1].strip())
        refs.append(sample["expected_output"])
        times.append(latency)

    metrics = compute_metrics(preds, refs)
    metrics["avg_latency"] = sum(times) / len(times)
    return metrics

if __name__ == "__main__":
    dataset = load_eval_dataset()
    base_metrics = run_eval("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dataset)
    ft_metrics = run_eval("merged_model", "merged_model", dataset)
    print("\n📊 Results:")
    print("Base Model:", base_metrics)
    print("Finetuned Model:", ft_metrics)

    with open("eval_results.json", "w") as f:
        json.dump({"base": base_metrics, "finetuned": ft_metrics}, f, indent=2)