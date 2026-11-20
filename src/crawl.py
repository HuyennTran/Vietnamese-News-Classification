"""Collect Vietnamese news headlines from RSS into data/raw/<source>.jsonl.

Run: python src/crawl.py --source tuoitre --max-per-class 125
Requires: pip install requests
"""

import argparse
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests


FEEDS = {
    "vnexpress": {
        "education": "https://vnexpress.net/rss/giao-duc.rss",
        "business": "https://vnexpress.net/rss/kinh-doanh.rss",
        "sports": "https://vnexpress.net/rss/the-thao.rss",
        "health": "https://vnexpress.net/rss/suc-khoe.rss",
    },
    "dantri": {
        "education": "https://dantri.com.vn/rss/giao-duc.rss",
        "business": "https://dantri.com.vn/rss/kinh-doanh.rss",
        "sports": "https://dantri.com.vn/rss/the-thao.rss",
        "health": "https://dantri.com.vn/rss/suc-khoe.rss",
    },
    "tuoitre": {
        "education": "https://tuoitre.vn/rss/giao-duc.rss",
        "business": "https://tuoitre.vn/rss/kinh-doanh.rss",
        "sports": "https://tuoitre.vn/rss/the-thao.rss",
        "health": "https://tuoitre.vn/rss/suc-khoe.rss",
    },
    "thanhnien": {
        "education": "https://thanhnien.vn/rss/giao-duc.rss",
        "business": "https://thanhnien.vn/rss/kinh-te.rss",
        "sports": "https://thanhnien.vn/rss/the-thao.rss",
        "health": "https://thanhnien.vn/rss/suc-khoe.rss",
    },
}
LABELS = {"education": "Giáo dục", "business": "Kinh doanh",
          "sports": "Thể thao", "health": "Sức khỏe"}
ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "src" else ROOT


def normalize_url(url):
    p = urlsplit(url.strip())
    return urlunsplit((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", ""))


def field(node, name):
    for child in node:
        if child.tag.split("}")[-1] == name:
            return "".join(child.itertext()).strip()
    return ""


def parse_date(value):
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (ValueError, TypeError, IndexError):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return ""


def parse_rss(content):
    for item in ET.fromstring(content).iter():
        if item.tag.split("}")[-1] != "item":
            continue
        title = re.sub(r"<[^>]+>", "", unescape(field(item, "title"))).strip()
        url = normalize_url(field(item, "link"))
        date = field(item, "pubDate") or field(item, "date") or field(item, "published")
        if title and url:
            yield title, url, parse_date(date)


def crawl(source, limit):
    path = ROOT / "data" / "raw" / f"{source}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    seen = {normalize_url(x["url"]) for x in old}
    counts = Counter(x["category"] for x in old)
    print(f"\n{source}: existing {len(old)} articles", flush=True)

    with path.open("a", encoding="utf-8") as out:
        for topic, rss_url in FEEDS[source].items():
            category = LABELS[topic]
            if counts[category] >= limit:
                continue
            try:
                response = requests.get(rss_url, timeout=15, headers={"User-Agent": "StudentNewsCrawler/1.0"})
                response.raise_for_status()
                items = list(parse_rss(response.content))
            except (requests.RequestException, ET.ParseError) as exc:
                print(f"  {topic}: ERROR {exc}", flush=True)
                continue

            added = 0
            for title, url, publish_date in items:
                if counts[category] >= limit:
                    break
                host = urlsplit(url).hostname or ""
                domain = urlsplit(rss_url).hostname or ""
                if url in seen or (host != domain and not host.endswith("." + domain)):
                    continue
                article = {
                    "id": source + "_" + hashlib.sha256(url.encode()).hexdigest()[:16],
                    "source": source, "url": url, "category": category,
                    "title": title, "content": "", "publish_date": publish_date,
                }
                out.write(json.dumps(article, ensure_ascii=False) + "\n")
                seen.add(url)
                counts[category] += 1
                added += 1
            print(f"  {topic}: +{added}, total {counts[category]}/{limit} (RSS: {len(items)})", flush=True)
            time.sleep(1)
    print(f"  Saved to {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["all", *FEEDS], default="all")
    parser.add_argument("--max-per-class", type=int, default=125)
    args = parser.parse_args()
    if args.max_per_class < 1:
        parser.error("--max-per-class must be positive")
    for src in (FEEDS if args.source == "all" else [args.source]):
        crawl(src, args.max_per_class)