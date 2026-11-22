
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data"

LABELS_FILE = DATA_DIR / "labels.json"
REPORT_FILE = DATA_DIR / "preprocessing_report.json"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CATEGORY_MAPPING = {
    "giáo dục": "education",
    "kinh doanh": "business",
    "thể thao": "sports",
    "sức khỏe": "health",
}

LABEL_MAPPING = {
    "education": 0,
    "business": 1,
    "sports": 2,
    "health": 3,
}

OUTPUT_COLUMNS = [
    "id",
    "source",
    "url",
    "title",
    "content",
    "text",
    "publish_date",
    "category",
    "label",
]


def clean_text(value):
    if value is None:
        return ""

    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_category(category):
    if category is None:
        return None

    category = clean_text(category).casefold()

    return CATEGORY_MAPPING.get(category)


def normalize_date(date_value):
    if date_value is None or str(date_value).strip() == "":
        return ""

    try:
        date = pd.to_datetime(date_value, errors="coerce")

        if pd.isna(date):
            return ""

        return date.strftime("%Y-%m-%d")

    except (ValueError, TypeError):
        return ""


def normalize_url(url):
    if not url:
        return ""

    url = str(url).strip().lower()
    url = url.split("?")[0]
    url = url.split("#")[0]
    url = url.rstrip("/")

    return url


def load_raw_data():
    records = []
    source_stats = {}

    jsonl_files = sorted(RAW_DIR.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found in: {RAW_DIR}"
        )

    for file_path in jsonl_files:
        source_name = file_path.stem

        total_lines = 0
        valid_records = 0
        invalid_json = 0
        invalid_fields = 0

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                total_lines += 1

                try:
                    article = json.loads(line)

                except json.JSONDecodeError:
                    invalid_json += 1
                    continue

                if not isinstance(article, dict):
                    invalid_fields += 1
                    continue

                required_fields = [
                    "id",
                    "source",
                    "url",
                    "category",
                    "title",
                ]

                if any(
                    article.get(field) is None
                    or str(article.get(field)).strip() == ""
                    for field in required_fields
                ):
                    invalid_fields += 1
                    continue

                article["content"] = article.get("content") or ""
                article["publish_date"] = (
                    article.get("publish_date") or ""
                )

                records.append(article)
                valid_records += 1

        source_stats[source_name] = {
            "total_lines": total_lines,
            "valid_records": valid_records,
            "invalid_json": invalid_json,
            "invalid_fields": invalid_fields,
        }

    return records, source_stats


def save_labels():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(LABELS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            LABEL_MAPPING,
            file,
            ensure_ascii=False,
            indent=4,
        )


def preprocess_records(records):
    processed_records = []
    rejected_categories = 0

    for article in records:
        title = clean_text(article.get("title", ""))
        content = clean_text(article.get("content", ""))

        category_original = clean_text(
            article.get("category", "")
        )

        category = normalize_category(category_original)

        if category not in LABEL_MAPPING:
            rejected_categories += 1
            continue

        label = LABEL_MAPPING[category]

        if content:
            text = f"{title}. {content}".strip()
        else:
            text = title

        processed_records.append({
            "id": str(article.get("id", "")),
            "source": clean_text(article.get("source", "")),
            "url": str(article.get("url", "")).strip(),
            "title": title,
            "content": content,
            "text": text,
            "publish_date": normalize_date(
                article.get("publish_date")
            ),
            "category": category,
            "category_original": category_original,
            "label": label,
        })

    df = pd.DataFrame(processed_records)

    if df.empty:
        raise ValueError(
            "No valid records remain after preprocessing."
        )

    return df, rejected_categories


def remove_duplicates(df):
    initial_count = len(df)

    df = df.copy()

    df["normalized_url"] = df["url"].apply(normalize_url)

    has_url = df["normalized_url"] != ""

    url_duplicates = df.loc[has_url].duplicated(
        subset=["normalized_url"],
        keep="first",
    )

    duplicate_url_count = int(url_duplicates.sum())

    df = df.drop(
        df.loc[has_url].index[url_duplicates]
    )

    has_date = df["publish_date"] != ""

    title_date_duplicates = df.loc[has_date].duplicated(
        subset=["title", "publish_date"],
        keep="first",
    )

    duplicate_title_date_count = int(
        title_date_duplicates.sum()
    )

    df = df.drop(
        df.loc[has_date].index[title_date_duplicates]
    )

    df = df.drop(columns=["normalized_url"])

    removed_count = initial_count - len(df)

    return df, {
        "initial_records": initial_count,
        "duplicate_urls": duplicate_url_count,
        "duplicate_title_date": duplicate_title_date_count,
        "total_removed": removed_count,
        "remaining_records": len(df),
    }


def split_dataset(df, random_state=RANDOM_STATE):
    train_df, temporary_df = train_test_split(
        df,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=random_state,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temporary_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=random_state,
        stratify=temporary_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_dataset(df, output_path):
    df = df.copy()
    df = df[OUTPUT_COLUMNS]

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def get_dataset_statistics(df):
    return {
        "total_records": int(len(df)),
        "class_distribution": {
            str(key): int(value)
            for key, value in df["category"].value_counts().items()
        },
        "source_distribution": {
            str(key): int(value)
            for key, value in df["source"].value_counts().items()
        },
        "missing_content": int(
            (df["content"].str.strip() == "").sum()
        ),
        "missing_publish_date": int(
            (df["publish_date"].str.strip() == "").sum()
        ),
        "title_only_records": int(
            (
                (df["content"].str.strip() == "")
                & (df["title"].str.strip() != "")
            ).sum()
        ),
    }


def generate_report(
    source_stats,
    preprocessing_stats,
    duplicate_stats,
    rejected_categories,
    train_df,
    val_df,
    test_df,
):
    all_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    report = {
        "input": {
            "raw_directory": str(RAW_DIR),
            "source_statistics": source_stats,
        },
        "preprocessing": {
            "records_after_preprocessing": int(
                preprocessing_stats
            ),
            "rejected_categories": int(rejected_categories),
            "duplicate_removal": duplicate_stats,
        },
        "dataset": get_dataset_statistics(all_df),
        "split": {
            "random_state": RANDOM_STATE,
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "train_records": int(len(train_df)),
            "validation_records": int(len(val_df)),
            "test_records": int(len(test_df)),
        },
        "class_distribution_by_split": {
            "train": {
                str(k): int(v)
                for k, v in train_df["category"].value_counts().items()
            },
            "validation": {
                str(k): int(v)
                for k, v in val_df["category"].value_counts().items()
            },
            "test": {
                str(k): int(v)
                for k, v in test_df["category"].value_counts().items()
            },
        },
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_records, source_stats = load_raw_data()

    save_labels()

    df, rejected_categories = preprocess_records(raw_records)

    preprocessing_count = len(df)

    df, duplicate_stats = remove_duplicates(df)

    if len(df) == 0:
        raise ValueError(
            "No records remain after duplicate removal."
        )

    train_df, val_df, test_df = split_dataset(
        df,
        random_state=RANDOM_STATE,
    )

    save_dataset(train_df, TRAIN_FILE)
    save_dataset(val_df, VAL_FILE)
    save_dataset(test_df, TEST_FILE)

    generate_report(
        source_stats=source_stats,
        preprocessing_stats=preprocessing_count,
        duplicate_stats=duplicate_stats,
        rejected_categories=rejected_categories,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


if __name__ == "__main__":
    main()
