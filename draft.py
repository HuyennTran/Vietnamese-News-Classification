import json
from collections import Counter

path = "data/raw/tuoitre.jsonl"

with open(path, encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

urls = [x["url"] for x in data]
titles = [x["title"].strip().lower() for x in data]

print("Total:", len(data))
print("Classes:", dict(Counter(x["category"] for x in data)))
print("Duplicate URLs:", len(urls) - len(set(urls)))
print("Duplicate titles:", len(titles) - len(set(titles)))
print("Missing dates:", sum(not x["publish_date"] for x in data))