"""
情景故事生成器 — 把当天所有生词融进情景短文 + 对话。

词数自适应档位：
  1–4  词 → 迷你版：叙述 3 句 + 对话 4 行（1 个场景）
  5–10 词 → 标准版：叙述 5 句 + 对话 6–8 行（1 个场景）
  11–18词 → 加长版：叙述 6–7 句 + 对话 10–12 行（1 个场景）
  19+ 词  → 双场景：拆成 2 组，每组叙述 5 句 + 对话 6–8 行
"""

import json
import os
import re
import requests
from pathlib import Path


# ---------------------------------------------------------------------------
# 档位配置
# ---------------------------------------------------------------------------

def _tier(n: int) -> dict:
    """根据词数返回生成配置."""
    if n <= 4:
        return {
            "label": "迷你版",
            "scenes": 1,
            "narrative_sentences": "3",
            "dialogue_lines": "4",
        }
    elif n <= 10:
        return {
            "label": "标准版",
            "scenes": 1,
            "narrative_sentences": "5",
            "dialogue_lines": "6–8",
        }
    elif n <= 18:
        return {
            "label": "加长版",
            "scenes": 1,
            "narrative_sentences": "6–7",
            "dialogue_lines": "10–12",
        }
    else:
        return {
            "label": "双场景",
            "scenes": 2,
            "narrative_sentences": "5",
            "dialogue_lines": "6–8",
        }


# ---------------------------------------------------------------------------
# 单场景 prompt
# ---------------------------------------------------------------------------

_SINGLE_SCENE_SYSTEM = """\
You are an English learning content creator.
Given a list of vocabulary words, write a situational story for learners.

Requirements:
1. Choose ONE realistic everyday scene that allows ALL the given words to appear naturally and without feeling forced.
2. **Narrative section**: Write exactly {narrative_sentences} English sentences. Use EVERY target word at least once. **Bold** each target word inline. After each sentence, add its Chinese translation on the next line wrapped in （）.
3. **Dialogue section**: Write a natural {dialogue_lines}-line A/B conversation in the same scene. Use at least half the target words (bold them). After each speaker line, add Chinese translation in （）.
4. **词汇速查 section**: A markdown table | 单词 | 含义 | listing every word with a brief Chinese meaning.

Output ONLY the following markdown, no extra commentary:

# Day {day_num} — 情景练习（{label}）

> 场景：<one-line scene description in Chinese>

## 叙述段

<sentence 1 with **bolded** target words>
（<Chinese>）

<sentence 2>
（<Chinese>）

...

## 对话

**A:** <line>
（<Chinese>）

**B:** <line>
（<Chinese>）

...

## 词汇速查

| 单词 | 含义 |
|------|------|
| word | meaning |
"""

# ---------------------------------------------------------------------------
# 双场景 prompt（词数 19+）
# ---------------------------------------------------------------------------

_DUAL_SCENE_SYSTEM = """\
You are an English learning content creator.
Given a large list of vocabulary words, write TWO separate situational stories for learners. \
Split the words roughly in half between the two scenes so each scene feels natural.

For EACH scene:
1. Choose a different realistic everyday scene.
2. **Narrative**: 5 sentences using the assigned words (**bold** each). Chinese translation after each sentence in （）.
3. **Dialogue**: 6–8 line A/B conversation using at least half the assigned words (**bold**). Chinese after each line in （）.

After both scenes, write a combined **词汇速查** table listing ALL words.

Output ONLY the following markdown:

# Day {day_num} — 情景练习（双场景）

## 场景一

> 场景：<description in Chinese>

### 叙述段

<sentence>
（<Chinese>）

...

### 对话

**A:** ...
（<Chinese>）

**B:** ...
（<Chinese>）

...

## 场景二

> 场景：<description in Chinese>

### 叙述段

...

### 对话

...

## 词汇速查

| 单词 | 含义 |
|------|------|
| word | meaning |
"""


class StoryGenerator:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (
            os.getenv("QWEN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = os.getenv("QWEN_MODEL") or "deepseek-v3.2"

    def _chat(self, system: str, user: str) -> str:
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
            "temperature": 0.7,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def generate(self, words: list[dict], day_num: int) -> str:
        """
        words: parse_study_md() 返回的列表，每项有 word / meaning / dialogues。
        返回生成的 Markdown 字符串。
        """
        vocab = [
            {"word": w["word"], "meaning": w["meaning"]}
            for w in words
            if w.get("word")
        ]
        n = len(vocab)

        if n == 0:
            return f"# Day {day_num} — 情景练习\n\n（没有词汇，无法生成故事）\n"

        cfg = _tier(n)
        print(f"  [STORY] {n} words → {cfg['label']} ({cfg['scenes']} scene(s))", flush=True)

        if cfg["scenes"] == 1:
            system_prompt = (
                _SINGLE_SCENE_SYSTEM
                .replace("{narrative_sentences}", cfg["narrative_sentences"])
                .replace("{dialogue_lines}", cfg["dialogue_lines"])
                .replace("{day_num}", str(day_num))
                .replace("{label}", cfg["label"])
            )
        else:
            system_prompt = _DUAL_SCENE_SYSTEM.replace("{day_num}", str(day_num))

        user_msg = json.dumps(
            {"day": day_num, "vocabulary": vocab}, ensure_ascii=False
        )

        try:
            result = self._chat(system_prompt, user_msg)
            if not result.strip().startswith("#"):
                result = f"# Day {day_num} — 情景练习\n\n{result}"
            return result
        except Exception as e:
            print(f"  [STORY] LLM error: {e}")
            return f"# Day {day_num} — 情景练习\n\n（生成失败：{e}）\n"

    def process_day(self, day_dir: Path, words: list[dict], day_num: int) -> Path:
        """生成故事并保存到 day_dir/day{n}_story.md，返回文件路径."""
        story_path = day_dir / f"day{day_num}_story.md"
        print(f"  [STORY] generating for {len(words)} words ...", flush=True)
        content = self.generate(words, day_num)
        story_path.write_text(content, encoding="utf-8")
        print(f"  [STORY] saved: {story_path}")
        return story_path


# ---------------------------------------------------------------------------
# CLI: python story_generator.py word/day5/day5_study.md
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python story_generator.py <study.md path>")
        sys.exit(1)

    md_path = Path(sys.argv[1])
    if not md_path.exists():
        print(f"File not found: {md_path}")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import parse_study_md

    text = md_path.read_text("utf-8")
    words = parse_study_md(text)

    m = re.search(r"day(\d+)", md_path.name)
    day_num = int(m.group(1)) if m else 1

    gen = StoryGenerator()
    out = gen.process_day(md_path.parent, words, day_num)
    print(f"Done: {out}")
