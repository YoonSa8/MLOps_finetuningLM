from prefect import flow
from pipelines.serving_pipeline import serving_pipeline
import requests
import os
import dotenv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

API_URL = "http://localhost:8000/generate_text"
AUTH_TOKEN = os.getenv("AUTH_TOKEN")


def call_api(prompt: str , max_new_token: int = 200, temperature: float = 0.7):
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    payload = {
        "prompt": prompt,
        "max_new_token": max_new_token,
        "temperature": temperature
    }
    try:
        res = requests.post(API_URL, json=payload, headers=headers)
        return res.json()["output"]
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")


def serving_pipeline(
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7):
    output = call_api(prompt, max_tokens, temperature)
    if output:
        print("Model Output:\n")
        print(output)


@flow
def deploy(prompt: str, max_tokens: int = 200, temperature: float = 0.7):
    serving_pipeline(prompt, max_tokens, temperature)


if __name__ == "__main__":
    deploy.deploy(
        name="my-bot",
        work_pool_name="my-work-pool",
        image="my-image",
        push=False,
        cron="* * * * *",
    )
