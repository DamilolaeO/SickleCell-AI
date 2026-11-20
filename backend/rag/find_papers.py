"""Find open-access (CC BY) reviews about sickle cell disease from Europe PMC.

This only LISTS candidates. You review them and mark which to include.
"""
import csv
import requests

QUERY = (
    'TITLE:"sickle cell" '
    'AND (TITLE:"review" OR TITLE:"meta-analysis") '
    'AND OPEN_ACCESS:y AND LICENSE:"cc by" '
    'AND PUB_YEAR:[2016 TO 2026]'
)
URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

params = {"query": QUERY, "format": "json", "resultType": "core", "pageSize": 100}
results = requests.get(URL, params=params, timeout=60).json()["resultList"]["result"]

with open("data/papers_candidates_2.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["include", "pmcid", "year", "title", "journal", "licence", "doi", "url", "abstract"])
    for r in results:
        journal = r.get("journalTitle") or r.get("journalInfo", {}).get("journal", {}).get("title", "")
        pmcid = r.get("pmcid", "")
        writer.writerow([
            "",
            pmcid,
            r.get("pubYear", ""),
            r.get("title", ""),
            journal,
            r.get("license", ""),
            r.get("doi", ""),
            f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else "",
            (r.get("abstractText", "") or "")[:600],
        ])

print(f"Found {len(results)} candidate papers. Open data/papers_candidates_2.csv")