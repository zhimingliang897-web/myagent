import json
import os
import re
from pathlib import Path

import pdfplumber
import requests



class PDFLargeWordExtractor:
    def __init__(self, pdf_folder="word"):
        self.pdf_folder = Path(pdf_folder)
        self.large_chars = []
        self.llm_api_key = os.getenv("DASHSCOPE_API_KEY") 
        self.llm_base_url = os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.llm_model = os.getenv("QWEN_MODEL") or "deepseek-v3.2"
        self.llm_mode = (os.getenv("LLM_MODE") or "auto").lower()
    
    def extract_large_font_chars(self, pdf_path, threshold=11):
        """提取最大字号的字符"""
        try:
            chars_by_size = {}

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    if hasattr(page, 'chars') and page.chars:
                        for obj in page.chars:
                            size = round(obj['size'], 1)
                            text = obj['text'].strip()

                            if text and size > 0:
                                if size not in chars_by_size:
                                    chars_by_size[size] = []
                                chars_by_size[size].append(text)

            # 找最大的字号
            if chars_by_size:
                max_size = max(chars_by_size.keys())
                large_chars = chars_by_size[max_size]

                print(f"找到最大字号: {max_size} ({len(large_chars)} 个字符)\n")
                return large_chars

            return []

        except Exception as e:
            print(f"[ERR] 读取失败: {e}")
            return []
    
    def interactive_combine(self, chars):
        """交互式拼合字符成单词"""
        if not chars:
            print("没有找到字符")
            return []
        
        print("=" * 70)
        print("字符列表 (从PDF最大字号提取):")
        print("=" * 70)
        
        # 显示所有字符
        display = ""
        for i, char in enumerate(chars):
            display += char
            if (i + 1) % 20 == 0:
                print(f"{i+1:3d}: {display}")
                display = ""
        if display:
            print(f"{len(chars):3d}: {display}")
        
        print(f"\n共 {len(chars)} 个字符\n")
        
        words = []
        print("="*70)
        print("输入方式: 输入单词包含的字符位置范围 (用逗号分隔)")
        print("例如: 0-6 (位置0到6拼成一个词)")
        print("      或: 0,1,2,3,4,5,6")
        print("输入 'done' 完成")
        print("="*70 + "\n")
        
        used_indices = set()
        
        while True:
            try:
                user_input = input("输入范围: ").strip().lower()
                
                if user_input == 'done':
                    break
                
                # 解析范围
                indices = set()
                
                if '-' in user_input:
                    # 处理范围 0-5
                    parts = user_input.split('-')
                    if len(parts) == 2:
                        start = int(parts[0].strip())
                        end = int(parts[1].strip())
                        indices = set(range(start, end + 1))
                else:
                    # 处理逗号分隔的索引 0,1,2,3,4
                    parts = user_input.split(',')
                    for p in parts:
                        indices.add(int(p.strip()))
                
                # 检查是否已使用
                overlap = indices & used_indices
                if overlap:
                    print(f"[WARN] 位置 {sorted(overlap)} 已被使用\n")
                    continue
                
                # 拼合字符
                word_chars = [chars[i] for i in sorted(indices)]
                word = ''.join(word_chars)
                
                # 确认
                print(f"拼合为: '{word}' (位置: {sorted(indices)})")
                confirm = input("确认? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    words.append(word)
                    used_indices.update(indices)
                    print(f"[OK] 已添加: {word}\n")
                    
                    # 显示已使用的索引
                    remaining = [i for i in range(len(chars)) if i not in used_indices]
                    if remaining:
                        print(f"剩余未使用位置: {remaining[:10]}{'...' if len(remaining) > 10 else ''}\n")
                else:
                    print("已取消\n")
            
            except (ValueError, IndexError) as e:
                print(f"[ERR] 输入有误: {e}\n")
                print("示例: 0-5 或 0,1,2,3,4,5\n")
        
        return words

    def _llm_ready(self):
        if not self.llm_api_key or not self.llm_base_url:
            return False
        return True

    def _build_chat_url(self):
        base = self.llm_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _parse_llm_words(self, text):
        text = text.strip()
        if not text:
            return []

        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("words"), list):
                return [str(w).strip() for w in data["words"] if str(w).strip()]
            if isinstance(data, list):
                return [str(w).strip() for w in data if str(w).strip()]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [str(w).strip() for w in data if str(w).strip()]
            except json.JSONDecodeError:
                pass

        parts = re.split(r"[,\n]+", text)
        return [p.strip() for p in parts if p.strip()]

    def combine_with_llm(self, chars):
        if not chars:
            return []
        if not self._llm_ready():
            print("LLM not configured; set DASHSCOPE_API_KEY.")
            return []

        sequence = "".join(chars)
        system_prompt = (
            "Split the given character sequence into words. "
            "Use all characters exactly once, keep order, and do not add or remove characters. "
            "Return JSON only: a list of words or {\"words\": [...]}."
        )
        user_prompt = f"Chars ({len(sequence)}): {sequence}"

        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                self._build_chat_url(),
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM request failed: {e}")
            return []

        words = self._parse_llm_words(content)
        if not words:
            return []

        if "".join(words) != sequence:
            print("LLM output does not match the original sequence exactly.")
        return words

    def combine_chars(self, chars):
        if self.llm_mode == "interactive":
            return self.interactive_combine(chars)
        if self.llm_mode in ("llm", "auto"):
            words = self.combine_with_llm(chars)
            if words:
                return words
            if self.llm_mode == "llm":
                return []
        return self.interactive_combine(chars)
    
    def save_words(self, words, output_file):
        """保存单词到指定路径"""
        if not words:
            print("\nNo words to save.")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(word + "\n")

        print(f"\n[OK] 已保存 {len(words)} 个单词到 {output_file}:")
        for i, word in enumerate(words, 1):
            print(f"  {i}. {word}")


def main():
    print("=" * 70)
    print("PDF -> words")
    print("=" * 70 + "\n")

    extractor = PDFLargeWordExtractor()

    # 扫描 word/day*/ 子目录中的 PDF
    day_dirs = sorted(extractor.pdf_folder.glob("day*"))
    if not day_dirs:
        print(f"[ERR] word/ 下没有 day* 目录")
        return

    for day_dir in day_dirs:
        if not day_dir.is_dir():
            continue

        pdf_files = sorted(day_dir.glob("*.pdf"))
        if not pdf_files:
            continue

        for pdf_file in pdf_files:
            txt_file = pdf_file.with_suffix(".txt")
            if txt_file.exists():
                print(f"[SKIP] {txt_file} already exists")
                continue

            print(f"processing: {pdf_file}")
            print("-" * 70)

            large_chars = extractor.extract_large_font_chars(str(pdf_file))
            if not large_chars:
                print("no chars found\n")
                continue

            words = extractor.combine_chars(large_chars)

            if words:
                extractor.save_words(words, txt_file)

            print()


if __name__ == "__main__":
    main()
