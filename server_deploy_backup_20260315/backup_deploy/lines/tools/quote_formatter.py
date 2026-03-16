import json
import os
import requests
import time
from pathlib import Path


class QuoteFormatter:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = os.getenv("QWEN_MODEL") or "deepseek-v3.2"

    def _chat(self, system, user):
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def format_word(self, word_data):
        """调 LLM 将一个词的台词整理成学习笔记"""
        word = word_data.get("phrase", "")
        results = word_data.get("results", [])
        if not results:
            return f"## {word}\n\n(no quotes found)\n"

        compact = []
        for r in results[:5]:
            title = r.get("title", "")
            year = r.get("year", "")
            quotes = r.get("quotes", [])
            if quotes:
                compact.append({
                    "title": f"{title} ({year})" if year else title,
                    "quotes": quotes[:4],
                })

        system_prompt = (
            "You are an English learning assistant. "
            "Given a vocabulary word and raw movie/TV quotes containing it, produce a study note.\n\n"
            "Format:\n\n"
            "## <word>\n\n"
            "**meaning:** <brief Chinese + English definition>\n\n"
            "**example dialogues:**\n\n"
            "Pick the 2-3 best quotes showing natural usage. "
            "Rewrite each as a short dialogue with context. Keep original English, add Chinese translation.\n\n"
            "Dialogue format:\n"
            "> **<Title> (<Year>)**\n"
            "> A: <line>\n"
            "> (<Chinese>)\n"
            "> B: <line>\n"
            "> (<Chinese>)\n\n"
            "Single sentence format:\n"
            "> **<Title> (<Year>)**\n"
            '> "<quote>"\n'
            "> (<Chinese>)\n\n"
            "Only output the study note."
        )

        user_msg = json.dumps({"word": word, "results": compact}, ensure_ascii=False)

        try:
            return self._chat(system_prompt, user_msg)
        except Exception as e:
            print(f"  [ERR] LLM failed for '{word}': {e}")
            return f"## {word}\n\n(LLM error: {e})\n"

    def process_file(self, json_path):
        """处理一个 JSON 结果文件，生成学习材料 Markdown"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stem = json_path.stem
        output_path = json_path.parent / f"{stem}_study.md"

        sections = []
        total = len(data)

        for idx, word_data in enumerate(data, 1):
            word = word_data.get("phrase", "?")
            print(f"  [{idx}/{total}] {word} ... ", end="", flush=True)

            section = self.format_word(word_data)
            sections.append(section)
            print("done")

            if idx < total:
                time.sleep(1)

        content = f"# {stem} - study notes\n\n" + "\n\n---\n\n".join(sections) + "\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[OK] saved: {output_path}")
        return output_path


def main():
    print("=" * 60)
    print("Quote Formatter -- generate study notes")
    print("=" * 60)

    fmt = QuoteFormatter()
    word_dir = Path("word")

    day_dirs = sorted(word_dir.glob("day*"))
    if not day_dirs:
        print("[ERR] no day* dirs in word/")
        return

    for day_dir in day_dirs:
        if not day_dir.is_dir():
            continue

        json_files = sorted(day_dir.glob("*.json"))
        for jf in json_files:
            study_file = jf.parent / f"{jf.stem}_study.md"
            if study_file.exists():
                print(f"[SKIP] {study_file} already exists")
                continue

            print(f"\nProcessing: {jf}")
            print("-" * 60)
            fmt.process_file(jf)

    print("\nDone.")


if __name__ == "__main__":
    main()
