"""
TTS 音频生成器 — 用 edge-tts 为单词和台词生成 mp3 文件。

音频目录结构（word/day5/audio/）:
  w_0.mp3              单词 0 发音
  w_1.mp3              单词 1 发音
  d_0_0_0.mp3          单词 0, 对话块 0, 台词行 0
  d_0_0_1.mp3          单词 0, 对话块 0, 台词行 1
  ...
"""

import asyncio
import re
import sys
from pathlib import Path

import edge_tts

VOICE = "en-US-AriaNeural"


def _is_speakable(text: str) -> bool:
    """判断文本是否值得生成语音（过滤纯符号 / 分隔线等）."""
    cleaned = re.sub(r"[^a-zA-Z]", "", text)
    return len(cleaned) >= 2


async def _generate_one(text: str, output: Path, voice: str = VOICE):
    """生成单条音频."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output))
    # 清理空文件（edge-tts 对某些输入会写出 0 字节文件）
    if output.exists() and output.stat().st_size == 0:
        output.unlink()


async def _generate_all(words: list[dict], audio_dir: Path, voice: str = VOICE):
    """为所有单词和台词行生成音频."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    tasks = []

    for wi, w in enumerate(words):
        # 单词发音
        word_file = audio_dir / f"w_{wi}.mp3"
        if not word_file.exists():
            tasks.append((w["word"], word_file))

        # 台词行
        for di, dlg in enumerate(w.get("dialogues", [])):
            for li, line in enumerate(dlg.get("lines", [])):
                en = line.get("en", "")
                if en and _is_speakable(en):
                    line_file = audio_dir / f"d_{wi}_{di}_{li}.mp3"
                    if not line_file.exists():
                        tasks.append((en, line_file))

    total = len(tasks)
    if total == 0:
        print("  [TTS] all audio files already exist, skip")
        return

    print(f"  [TTS] generating {total} audio files ...")
    for idx, (text, path) in enumerate(tasks, 1):
        try:
            await _generate_one(text, path, voice)
            if idx % 10 == 0 or idx == total:
                print(f"  [TTS] {idx}/{total}")
        except Exception as e:
            print(f"  [TTS] error on {path.name}: {e}")

    print(f"  [TTS] done — {total} files saved to {audio_dir}")


def generate(words: list[dict], audio_dir):
    """同步入口，供 pipeline 调用."""
    audio_dir = Path(audio_dir)
    asyncio.run(_generate_all(words, audio_dir))


# ---------------------------------------------------------------------------
# CLI: python tts_generator.py word/day5/day5_study.md
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tts_generator.py <study.md path>")
        sys.exit(1)

    md_path = Path(sys.argv[1])
    if not md_path.exists():
        print(f"File not found: {md_path}")
        sys.exit(1)

    # 复用 app.py 的解析函数
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from app import parse_study_md

    text = md_path.read_text("utf-8")
    words = parse_study_md(text)
    audio_dir = md_path.parent / "audio"
    generate(words, audio_dir)
