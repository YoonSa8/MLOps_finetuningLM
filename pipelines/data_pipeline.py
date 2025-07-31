from data_collection.pdf_extract import extract_pdf_dir
from data_collection.web_scrapper import scrape_urls
from data_collection.metadata import enrich_metadata
from data_collection.merge_data import merge_files
from preprocessing.preprocess import preprocess
import logging
from pathlib import Path
import yaml

CONFIG_PATH = Path("config/config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def collect_data_pipeline():
    pdf_dir = config["data_dir"]+"/pdf"
    pdf_out = config["data_dir"]+"/artifacts/raw_data/pdf_data.jsonl"
    extract_pdf_dir(pdf_dir, pdf_out)

    url_list = config.get("url_list", [])
    web_out = config["data_dir"]+"/artifacts/raw_data/web_data.jsonl"
    if url_list:
        scrape_urls(url_list, web_out)
    else:
        print("error extracting web data ....")

    domain_keywords = config.get("domain_keywords", [])
    web_enrich = config["data_dir"] + \
        "/artifacts/processed/web_data_enriched.jsonl"
    pdf_enrich = config["data_dir"] + \
        "/artifacts/processed/pdf_data_enriched.jsonl"
    enrich_metadata(pdf_out, pdf_enrich, domain_keywords)
    enrich_metadata(web_out, web_enrich, domain_keywords)

    merged_out = config["data_dir"]+"/clean/cleaned.jsonl"
    merge_files([web_enrich, pdf_enrich], merged_out)
