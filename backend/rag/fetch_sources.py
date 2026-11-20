"""Download every source in data/sources.csv and save it as clean text in data/raw/."""
import csv, io, time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Sicklewise prototype (patient education research)"}


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.body
    for tag in main(["script", "style", "nav", "footer", "aside", "form", "header"]):
        tag.decompose()
    return main.get_text("\n", strip=True)


def pdf_to_text(data):
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


with open("data/sources.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            resp = requests.get(row["url"], headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"FAILED {row['id']}: {e}")
            continue

        text = pdf_to_text(resp.content) if row["type"] == "pdf" else html_to_text(resp.text)

        header = (
            f"TITLE: {row['title']}\n"
            f"URL: {row['url']}\n"
            f"OWNER: {row['owner']}\n"
            f"LICENCE: {row['licence']}\n"
            f"POPULATION: {row['population']}\n"
            f"DOWNLOADED: {date.today()}\n\n"
        )
        (OUT / f"{row['id']}.txt").write_text(header + text, encoding="utf-8")
        print(f"Saved {row['id']} ({len(text)} characters)")
        time.sleep(2)  # be polite to the servers
