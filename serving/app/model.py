from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.config import MODEL_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR,
                                             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.eval()


def generate_answer(prompt: str, max_new_token=200, temperature=0.7):
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    print("generate answer")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_token,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    print("answer generated")
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    if "<|assistant|>" in result:
        clean_output = result.split("<|assistant|>")[-1].strip()
    else:
        clean_output = result.strip()
    print("Generated result:", result)
    return clean_output
