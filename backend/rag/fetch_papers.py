"""Download the full text of every paper marked Y in data/papers_candidates.csv.

Saves each paper as clean text in data/raw/, with a header that records the
authors, journal, year, DOI and licence so the app can credit it properly.
Papers already downloaded are skipped, so it is safe to run again.
"""
import csv, time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import requests

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Admin sections that add noise, not patient information
SKIP = ("author", "funding", "availability", "ethics", "competing", "conflict",
        "data availability", "consent", "acknowledg", "abbreviation", "supplementary")


def text_of(el):
    return " ".join(" ".join(el.itertext()).split()) if el is not None else ""


def parse(xml_bytes):
    root = ET.fromstring(xml_bytes)
    meta = root.find(".//article-meta")

    authors = []
    for c in meta.iter("contrib"):
        if c.get("contrib-type", "author") != "author":
            continue
        surname, given = c.findtext(".//surname"), c.findtext(".//given-names")
        if surname:
            authors.append(f"{given or ''} {surname}".strip())

    abstract = text_of(meta.find(".//abstract"))

    body_parts = []
    body = root.find(".//body")
    if body is not None:
        for sec in body.iter("sec"):
            title = (sec.findtext("title") or "").strip()
            if title.lower().startswith(SKIP):
                continue
            paras = [text_of(p) for p in sec.findall("p")]
            if paras:
                body_parts.append((title + "\n" if title else "") + "\n".join(paras))
    return authors, abstract, "\n\n".join(body_parts)


with open("data/papers_candidates.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["include"].strip().upper() != "Y":
            continue
        pmcid = row["pmcid"]
        if (OUT / f"paper-{pmcid}.txt").exists():
            continue  # already downloaded
        try:
            resp = requests.get(API.format(pmcid=pmcid), timeout=90)
            resp.raise_for_status()
            authors, abstract, body = parse(resp.content)
            note = "" if body else " (abstract only, full text not available)"
        except Exception as e:
            print(f"FAILED {pmcid}: {e}")
            time.sleep(3)
            continue

        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        if not author_str:
            author_str = "Authors not listed"
            print(f"  Warning: no authors found for {pmcid}, check the paper page")
        header = (
            f"TITLE: {row['title']}\n"
            f"URL: {row['url']}\n"
            f"OWNER: {author_str} ({row['journal']}, {row['year']})\n"
            f"DOI: {row['doi']}\n"
            f"LICENCE: {row['licence'].upper()}\n"
            f"TYPE: Research (label answers as 'Research suggests...')\n"
            f"DOWNLOADED: {date.today()}\n\n"
        )
        text = "ABSTRACT\n" + abstract + ("\n\n" + body if body else "")
        (OUT / f"paper-{pmcid}.txt").write_text(header + text, encoding="utf-8")
        print(f"Saved {pmcid}{note} ({len(text)} characters)")
        time.sleep(3)