"""Download a 100-document, open-access PMC corpus for bulbar MND research."""

from pathlib import Path
import time
import xml.etree.ElementTree as ET

import requests
import truststore

truststore.inject_into_ssl()


DATA_DIR = Path("data/pmc_xml")
EXCLUDED_DIR = Path("data/review_excluded")
TARGET_DOCUMENTS = 100
PAGE_SIZE = 100

# This search matches the same disease and bulbar-symptom criteria used below.
QUERY = (
    '("Amyotrophic Lateral Sclerosis"[Title/Abstract] '
    'OR "Motor Neuron Disease"[Title/Abstract] '
    'OR "Motor Neurone Disease"[Title/Abstract] '
    'OR "Progressive Bulbar Palsy"[Title/Abstract]) '
    'AND (bulbar[Title/Abstract] OR dysphagia[Title/Abstract] '
    'OR dysarthria[Title/Abstract] OR swallowing[Title/Abstract] '
    'OR tongue[Title/Abstract]) AND "open access"[filter]'
)

DISEASE_TERMS = (
    "amyotrophic lateral sclerosis",
    "motor neuron disease",
    "motor neurone disease",
    "progressive bulbar palsy",
)
BULBAR_TERMS = ("bulbar", "dysphagia", "dysarthria", "tongue", "speech", "swallowing")


def fetch_pmc_ids(query: str, retstart: int, retmax: int) -> list[str]:
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pmc", "term": query, "retstart": retstart, "retmax": retmax, "retmode": "json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["esearchresult"]["idlist"]


def is_relevant_article(xml_path: Path) -> bool:
    """Keep only articles whose title or abstract has both required concepts."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return False

    title = " ".join(" ".join(element.itertext()) for element in root.findall(".//article-title"))
    abstracts = " ".join(" ".join(element.itertext()) for element in root.findall(".//abstract"))
    text = f"{title} {abstracts}".lower()
    return any(term in text for term in DISEASE_TERMS) and any(term in text for term in BULBAR_TERMS)


def download_pmc_xml(pmc_id: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / f"PMC{pmc_id}.xml"
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pmc", "id": pmc_id, "retmode": "xml"},
        timeout=30,
    )
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")
    return destination


def move_to_review(xml_path: Path) -> None:
    EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)
    xml_path.replace(EXCLUDED_DIR / xml_path.name)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = [path for path in DATA_DIR.glob("PMC*.xml") if is_relevant_article(path)]
    if len(existing) >= TARGET_DOCUMENTS:
        print(f"Corpus already contains {len(existing)} eligible documents.")
        return

    known_ids = {
        path.stem.removeprefix("PMC")
        for folder in (DATA_DIR, EXCLUDED_DIR)
        for path in folder.glob("PMC*.xml")
    }
    print(f"Eligible documents already available: {len(existing)}. Target: {TARGET_DOCUMENTS}.")

    for retstart in range(0, 1_000, PAGE_SIZE):
        pmc_ids = fetch_pmc_ids(QUERY, retstart, PAGE_SIZE)
        if not pmc_ids:
            break
        for pmc_id in pmc_ids:
            if pmc_id in known_ids:
                continue
            xml_path = download_pmc_xml(pmc_id)
            known_ids.add(pmc_id)
            if is_relevant_article(xml_path):
                existing.append(xml_path)
                print(f"Kept {xml_path.name} ({len(existing)}/{TARGET_DOCUMENTS})")
            else:
                move_to_review(xml_path)
                print(f"Moved PMC{pmc_id}.xml to review_excluded")
            if len(existing) >= TARGET_DOCUMENTS:
                print("Target reached.")
                return
            time.sleep(0.34)  # Respect NCBI's unauthenticated request rate.

    print(f"Finished with {len(existing)} eligible documents; the search had no more candidates.")


if __name__ == "__main__":
    main()
