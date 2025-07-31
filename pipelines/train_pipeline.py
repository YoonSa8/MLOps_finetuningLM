from training.trainer import train_model
from pathlib import Path
import yaml
import logging

CONFIG_PATH = Path("config/config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def training_pipeline():
    model_name = config["base_model"]
    dataset_path = config["data_dir"]+"/qa/qa_formatted.jsonl"
    output_dir = config["save_dir"]
    train_model(model_name, dataset_path, output_dir)