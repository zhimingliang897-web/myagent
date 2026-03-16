import requests
import json
import csv
import time
from pathlib import Path


class PopMysticScraper:
    """
    通过 popmystic.com 的后端 OpenSearch API 查询台词。

    原理:
      popmystic.com 是一个 Vue SPA (单页应用)，前端页面通过 JS 动态渲染。
      直接用 requests.get 访问网页 URL 只能拿到空壳 HTML，看不到搜索结果。
      通过浏览器 DevTools (F12 -> Network) 抓包发现，前端实际调用的是:
        POST https://pop-opensearch-api-myxi6.ondigitalocean.app/search-scroll
      请求体是 OpenSearch/Elasticsearch 格式的 JSON query。
      所以我们直接调这个 API，绕过前端渲染，速度快且稳定。
    """

    def __init__(self):
        self.api_url = "https://pop-opensearch-api-myxi6.ondigitalocean.app/search-scroll"
        self.headers = {
            'Content-Type': 'application/json',
            'Origin': 'https://popmystic.com',
            'Referer': 'https://popmystic.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }

    def search_phrase(self, phrase):
        """搜索关键词，返回包含该词的影视作品和台词"""
        payload = {
            "query": {
                "bool": {
                    "must": {
                        "nested": {
                            "score_mode": "max",
                            "path": "phrase",
                            "inner_hits": {"size": 100},
                            "query": {
                                "match": {"phrase.text": phrase}
                            }
                        }
                    }
                }
            },
            "sort": ["_score", {"imdbId": "asc"}],
            "_source": True,
        }

        try:
            resp = requests.post(
                self.api_url, headers=self.headers,
                json=payload, timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e), "results": []}

        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)

        results = []
        for hit in hits:
            src = hit.get("_source", {})
            inner = (hit.get("inner_hits", {})
                         .get("phrase", {})
                         .get("hits", {})
                         .get("hits", []))
            quotes = [q["_source"]["text"] for q in inner if q.get("_source", {}).get("text")]

            results.append({
                "title": src.get("title", ""),
                "year": src.get("year", ""),
                "genre": src.get("genre", ""),
                "episode_title": src.get("episodeTitle", ""),
                "season": src.get("season", ""),
                "episode": src.get("episodeNumber", ""),
                "imdb_id": src.get("imdbId", ""),
                "quotes": quotes,
            })

        return {"status": "success", "phrase": phrase, "total": total, "results": results}

    def batch_search(self, words, delay=1):
        """批量查询，返回所有结果"""
        all_results = []
        total = len(words)
        print(f"\n-- batch: {total} words --\n")

        for idx, word in enumerate(words, 1):
            print(f"[{idx}/{total}] {word} ... ", end="", flush=True)
            result = self.search_phrase(word)

            if result["status"] == "success":
                print(f"found {result['total']} hits ({len(result['results'])} shows)")
            else:
                print(f"ERROR: {result['message']}")

            all_results.append(result)
            if idx < total:
                time.sleep(delay)

        return all_results


def save_json(results, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON saved: {filepath}")


def save_csv(results, filepath):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "total_hits", "title", "year", "genre",
                         "episode_title", "season", "episode", "imdb_id", "quote"])
        for r in results:
            word = r.get("phrase", "")
            total_hits = r.get("total", 0)
            for item in r.get("results", []):
                quote_text = " | ".join(item["quotes"]) if item["quotes"] else ""
                writer.writerow([
                    word, total_hits,
                    item["title"], item["year"], item["genre"],
                    item["episode_title"], item["season"], item["episode"],
                    item["imdb_id"], quote_text,
                ])
    print(f"[OK] CSV saved: {filepath}")


def load_words(filepath):
    p = Path(filepath)
    if not p.exists():
        print(f"[ERR] file not found: {filepath}")
        return []
    words = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[OK] loaded {len(words)} words from {filepath}")
    return words


def main():
    print("=" * 60)
    print("Pop Mystic -- batch search")
    print("=" * 60)

    scraper = PopMysticScraper()
    word_dir = Path("word")

    # 扫描 word/day*/ 子目录
    day_dirs = sorted(word_dir.glob("day*"))
    if not day_dirs:
        print("[ERR] no day* dirs in word/")
        return

    for day_dir in day_dirs:
        if not day_dir.is_dir():
            continue

        txt_files = sorted(day_dir.glob("*.txt"))
        if not txt_files:
            continue

        for txt_file in txt_files:
            stem = txt_file.stem
            json_out = day_dir / f"{stem}.json"

            if json_out.exists():
                print(f"[SKIP] {json_out} already exists")
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {txt_file}")
            print(f"{'='*60}")

            words = load_words(txt_file)
            if not words:
                continue

            results = scraper.batch_search(words)

            save_json(results, json_out)
            save_csv(results, day_dir / f"{stem}.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
