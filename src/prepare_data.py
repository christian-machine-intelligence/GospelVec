"""
Download KJV Gospel texts from bible-api.com.

Fetches Matthew, Mark, Luke, and John chapter by chapter
and saves raw text for activation extraction.

Usage:
  python src/prepare_data.py
"""

import re
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "gospels"

GOSPELS = {
    "matthew": ("Matthew", 28),
    "mark": ("Mark", 16),
    "luke": ("Luke", 24),
    "john": ("John", 21),
}


def fetch_gospel(book_name: str, n_chapters: int) -> str:
    """Fetch full Gospel text from bible-api.com (KJV)."""
    full_text = []
    for ch in range(1, n_chapters + 1):
        url = f"https://bible-api.com/{book_name}+{ch}?translation=kjv"
        print(f"  Chapter {ch}/{n_chapters} ...", end=" ", flush=True)
        for attempt in range(5):
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"rate limited, waiting {wait}s ...", end=" ", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        full_text.append(data["text"].strip())
        print("OK")
        time.sleep(0.5)
    return "\n\n".join(full_text)


def clean_text(text: str) -> str:
    """Light cleaning of KJV text."""
    text = re.sub(r"^\d+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for key, (book_name, n_chapters) in GOSPELS.items():
        out_path = DATA_DIR / f"{key}_raw.txt"
        if out_path.exists():
            print(f"{key}: already downloaded, skipping.")
            continue

        print(f"\n--- {book_name} ---")
        text = fetch_gospel(book_name, n_chapters)
        text = clean_text(text)
        out_path.write_text(text)
        print(f"  Saved: {out_path} ({len(text.split())} words)")

    print("\nDone.")


if __name__ == "__main__":
    main()
