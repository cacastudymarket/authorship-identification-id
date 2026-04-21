import requests
import pandas as pd
import time
import os
from datetime import datetime

os.makedirs("data/raw", exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 AuthorshipResearch/1.0 (student project)"
}

CATEGORIES = {
    "Politik": "Kategori:Politik_Indonesia",
    "Ekonomi": "Kategori:Ekonomi_Indonesia",
    "Olahraga": "Kategori:Olahraga_di_Indonesia",
    "Sains": "Kategori:Sejarah_Indonesia",
    "Hukum": "Kategori:Pendidikan_di_Indonesia",
}

BASE_URL = "https://id.wikipedia.org/w/api.php"


def get_articles_from_category(category: str, max_articles: int = 50) -> list:
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": max_articles,
        "cmtype": "page",
        "format": "json",
    }
    try:
        resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("categorymembers", [])
        return [(p["pageid"], p["title"]) for p in pages]
    except Exception as e:
        print(f"[ERROR] Gagal ambil kategori {category}: {e}")
        return []


def get_article_text(pageid: int) -> str:
    params = {
        "action": "query",
        "pageids": pageid,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    try:
        resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        text = pages.get(str(pageid), {}).get("extract", "")
        return text
    except Exception as e:
        print(f"[ERROR] Gagal ambil artikel {pageid}: {e}")
        return ""


def run_scraper():
    all_articles = []

    for label, category in CATEGORIES.items():
        print(f"\n[INFO] Mengambil kategori: {label}...")
        article_list = get_articles_from_category(category, max_articles=60)
        print(f"[INFO] Ditemukan {len(article_list)} artikel")

        for pageid, title in article_list:
            text = get_article_text(pageid)
            if len(text.split()) < 100:
                continue

            all_articles.append({
                "author": label,
                "title": title,
                "text": text,
                "word_count": len(text.split()),
                "scraped_at": datetime.now().isoformat(),
            })
            print(f"  [OK] {title[:60]}")
            time.sleep(0.5)

    df = pd.DataFrame(all_articles)

    if df.empty:
        print("[WARNING] Tidak ada artikel yang berhasil di-scrape.")
        return df

    print(f"\n[DONE] Total artikel: {len(df)}")
    print(df["author"].value_counts().to_string())

    df.to_csv("data/raw/wikipedia_id.csv", index=False, encoding="utf-8-sig")
    print("\n[SAVED] data/raw/wikipedia_id.csv")
    return df


if __name__ == "__main__":
    run_scraper()