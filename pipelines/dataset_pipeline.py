from dataset.qa_generate import generate_qa_from_text
from dataset.formatter import format_for_sft
from pathlib import Path
import yaml
import logging

CONFIG_PATH = Path("config/config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_dataset_pipeling():
    chunked_path = config["data_dir"]+"/clean/chunked.jsonl"
    qa_out = config["data_dir"]+"/qa/qa_together.jsonl"
    generate_qa_from_text(chunked_path, qa_out)
    formated_path = config["data_dir"]+"/qa/qa_formatted.jsonl"
    format_for_sft(qa_out, formated_path)