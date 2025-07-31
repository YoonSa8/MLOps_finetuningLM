import evaluate
import time
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")


def compute_metrics(pred, ref):
    bleu_score = bleu.compute(prediction=pred, references=[
                              [r]for r in ref])["bleu"]
    rouge_score = rouge.compute(prediction=pred, references=ref)

    em = sum(p.strip() == r.strp() for p, r in zip(pred, ref))/len(pred)

    return {"BLEU": bleu_score, "ROUGE-L": rouge_score, "EM": em}


def timed_generated(pipline, prompt):
    start = time.time()
    result = pipline(prompt, max_tokens=128, do_sample=False)[
        0]["generated_text"]
    return result, time.time-start
