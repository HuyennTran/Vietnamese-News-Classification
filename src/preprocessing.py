
"""
Preprocessing pipeline for the Vietnamese News Classification project.

This module loads and merges raw news datasets, cleans text, normalizes
categories, encodes labels, removes duplicates, and splits the data into
training, validation, and testing sets. It also exports processed CSV
files, label mappings, and a preprocessing report.

Run: python src/preprocessing.py
Requires: pip install pandas scikit-learn
"""

import json
import re
import html
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data"

CATEGORY_TO_LABEL = {
    "giáo dục": 0,
    "kinh doanh": 1,
    "thể thao": 2,
    "sức khỏe": 3
}

RANDOM_STATE = 42


def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def load_raw_data():
    records = []

    for file in sorted(RAW_DIR.glob("*.jsonl")):
        with open(file, "r", encoding="utf-8") as f:
            records.extend(
                json.loads(line)
                for line in f
                if line.strip()
            )

    return records


def preprocess_data(records):
    data = []

    for record in records:
        title = clean_text(record["title"])
        category = str(record["category"]).strip()

        if not title:
            continue

        date = pd.to_datetime(
            record.get("publish_date", ""),
            errors="coerce"
        )

        data.append({
            "id": record["id"],
            "source": record["source"],
            "url": record["url"],
            "title": title,
            "text": title,
            "publish_date": date.strftime("%Y-%m-%d") if pd.notna(date) else "",
            "category": category,
            "label": CATEGORY_TO_LABEL[category.casefold()]
        })

    return pd.DataFrame(data)


def remove_duplicates(df):
    before = len(df)

    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.drop_duplicates(subset=["title"], keep="first")
    df = df.reset_index(drop=True)

    return df, before - len(df)


def split_dataset(df):
    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp["label"]
    )

    return train, validation, test


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_raw_data()
    df = preprocess_data(records)

    df, duplicates_removed = remove_duplicates(df)

    train, validation, test = split_dataset(df)

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
        encoding="utf-8-sig"
    )

    validation.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(CATEGORY_TO_LABEL, f, ensure_ascii=False, indent=4)

    report = {
        "total_raw_records": len(records),
        "total_after_preprocessing": len(df),
        "duplicate_method": "url_or_title",
        "duplicates_removed": duplicates_removed,
        "train_size": len(train),
        "validation_size": len(validation),
        "test_size": len(test),
        "label_distribution": {
            str(k): int(v)
            for k, v in df["label"].value_counts().sort_index().items()
        }
    }

    with open(
        OUTPUT_DIR / "preprocessing_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
