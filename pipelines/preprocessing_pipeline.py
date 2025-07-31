from preprocessing.preprocess import preprocess
from pathlib import Path
import yaml
import logging

CONFIG_PATH = Path("config/config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def preprocessing_pipeline():
    cleaned_data = config["data_dir"]+"/clean/cleaned.jsonl"
    chunked_data = config["data_dir"]+"/clean/chunked.jsonl"
    preprocess(cleaned_data, chunked_data)
