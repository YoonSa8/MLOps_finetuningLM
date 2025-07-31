import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}


def extract_text_from_url(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.content, 'html.parser')
        title = soup.title.string if soup.title else url
        pragraph = soup.find_all('p')
        text = "\n".join(p.get_text().strip()
                         for p in pragraph if len(p.get_text().strip()) > 40)

        return {
            "source": "web",
            "title": title,
            "url": url,
            "text": text,
            "metadata": {
                "domain": urlparse(url).netloc,
                "length": len(text)
            }
        }
    except Exception as e:
        print(f"Failed to extract text from {url}: {e}")
        return None


def scrape_urls(url_list, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_results = []
    for url in url_list:
        result = extract_text_from_url(url)
        if result and result["text"]:
            all_results.append(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f'scrapped {len(all_results)} pages to {output_path}')


if __name__ == "__main__":
    urls = [
        "https://afdc.energy.gov/fuels/electricity_infrastructure.html",
        "https://www.energy.gov/eere/vehicles/articles/federal-incentives-electric-vehicles",
        "https://www.transportation.gov/rural/ev/toolkit"
    ]
    scrape_urls(urls, "artifacts/raw_data/web_data.jsonl")
