from evaluation.evaluate_model import run_eval, load_eval_dataset
import logging
import json 

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evaluation_pipeline():
    dataset = load_eval_dataset()
    base_metrics = run_eval("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dataset)
    ft_metrics = run_eval("merged_model", "merged_model", dataset)
    print("Results:")
    print("Base Model:", base_metrics)
    print("Finetuned Model:", ft_metrics)
    with open("eval_results.json", "w") as f:
        json.dump({"base": base_metrics, "finetuned": ft_metrics}, f, indent=2)