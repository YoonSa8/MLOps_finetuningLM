import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import json
from pathlib import Path
from .tracker import init_tracking


def load_jsonl_dataset(path):
    with open(path, "r", encoding="utf-8")as f:
        return [json.loads(line) for line in f]


def train_model(model_name: str, dataset_path: str,
                output_dir: str = "./saved_models", use_lora: bool = True, max_steps=100):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # add stability

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    if use_lora:
        config = LoraConfig(
            r=8, lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)

    data = load_jsonl_dataset(dataset_path)
    dataset = Dataset.from_list(data)

    def tokenize(example):
        prompt = example["instruction"] + "\n" + example["input"]
        response = example["output"]
        full_prompt = f"{prompt}\n{response}"

        tokenized = tokenizer(
            full_prompt,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(tokenize, batched=False,
                            remove_columns=dataset.column_names)

    print(tokenized[0])

    init_tracking(projrct="finetune_shortlm", run_name="qlora-run")
    collector = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        num_train_epochs=3,
        max_steps=max_steps,
        fp16=True, report_to="wandb",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=collector
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f'done fine tuning saved in {output_dir}')


if __name__ == "main":
    train_model()
