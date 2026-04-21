import pandas as pd
import re
import os

os.makedirs("data/processed", exist_ok=True)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"==.*?==", "", text)       # hapus judul section Wikipedia
    text = re.sub(r"\[\d+\]", "", text)        # hapus referensi [1], [2], dst
    text = re.sub(r"[^\w\s.,!?;:()\"\'-]", " ", text)
    text = text.strip()
    return text

def extract_features(text: str) -> dict:
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    avg_sent_len = len(words) / len(sentences) if sentences else 0
    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / len(words) if words else 0
    punct_count = len(re.findall(r"[.,!?;:]", text))
    punct_rate = (punct_count / len(words) * 100) if words else 0

    function_words = [
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan",
        "untuk", "pada", "adalah", "atau", "juga", "dalam", "tidak",
        "sebagai", "oleh", "bahwa", "karena", "jika", "tetapi", "namun",
        "seperti", "sudah", "akan", "bisa", "harus", "dapat", "lebih",
    ]
    words_lower = [w.lower() for w in words]
    fw_rate = {f"fw_{fw}": (words_lower.count(fw) / len(words) * 100) for fw in function_words}

    features = {
        "avg_word_length": round(avg_word_len, 4),
        "avg_sentence_length": round(avg_sent_len, 4),
        "type_token_ratio": round(ttr, 4),
        "punctuation_rate": round(punct_rate, 4),
        "total_words": len(words),
        "total_sentences": len(sentences),
        "total_unique_words": len(unique_words),
    }
    features.update(fw_rate)
    return features

def run_preprocessing():
    print("[INFO] Loading dataset...")
    df = pd.read_csv("data/raw/wikipedia_id.csv", encoding="utf-8-sig")
    print(f"[INFO] Total artikel: {len(df)}")

    print("[INFO] Cleaning teks...")
    df["text_clean"] = df["text"].apply(clean_text)

    # Filter teks terlalu pendek
    df["word_count_clean"] = df["text_clean"].apply(lambda x: len(x.split()))
    df = df[df["word_count_clean"] >= 100].reset_index(drop=True)
    print(f"[INFO] Setelah filter: {len(df)} artikel")

    print("[INFO] Ekstrak fitur stylometry...")
    features_list = df["text_clean"].apply(extract_features).tolist()
    features_df = pd.DataFrame(features_list)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # Simpan versi lengkap
    df.to_csv("data/processed/dataset_full.csv", index=False, encoding="utf-8-sig")

    # Simpan versi teks saja (untuk BERT nanti)
    df[["author", "title", "text_clean"]].to_csv(
        "data/processed/dataset_text.csv", index=False, encoding="utf-8-sig"
    )

    # Simpan versi fitur saja (untuk SVM)
    feature_cols = ["author"] + [c for c in df.columns if c.startswith(("avg_", "type_", "punct_", "total_", "fw_"))]
    df[feature_cols].to_csv(
        "data/processed/dataset_features.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n[DONE] Distribusi kelas:")
    print(df["author"].value_counts().to_string())
    print("\n[SAVED] data/processed/dataset_full.csv")
    print("[SAVED] data/processed/dataset_text.csv")
    print("[SAVED] data/processed/dataset_features.csv")

if __name__ == "__main__":
    run_preprocessing()