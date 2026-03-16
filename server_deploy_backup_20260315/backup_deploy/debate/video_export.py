"""
视频导出模块 v4.0 — 纯文字卡片 + 精确音视频同步

改动要点：
  1. 移除字幕烧录（发言卡片已含完整文字，字幕与卡片重叠问题彻底消除）
  2. 修复音视频同步：每张卡片独立对应精确音频段（含阶段卡静音）
  3. 全新卡片视觉设计：渐变背景 / 左侧色条 / 徽章标签 / 裁判内容优化
"""

import os
import re

from PIL import Image, ImageDraw, ImageFont

from config import OUTPUT_DIR, SPEECH_GAP
from tts_engine import get_audio_duration, ffmpeg_run


# ===== 全局常量 =====
VIDEO_W, VIDEO_H = 1280, 720
TITLE_DURATION    = 4.0   # 标题卡时长（秒）
PHASE_CARD_DURATION = 2.0 # 阶段过渡卡时长（秒）

_CN_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

# 阶段内部 key → 显示名（question/answer 合并为同一个阶段卡）
PHASE_DISPLAY_NAMES = {
    "opening": "开篇立论",
    "question": "攻辩质询",
    "answer":   "攻辩质询",
    "summary":  "攻辩小结",
    "free":     "自由辩论",
    "closing":  "总结陈词",
    "judge":    "裁判点评",
}


# ===== 工具函数 =====

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in _CN_FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """中文文本自动换行"""
    lines, current = [], ""
    for char in text:
        test = current + char
        w = draw.textbbox((0, 0), test, font=font)[2]
        if w > max_width:
            if current:
                lines.append(current)
            current = char
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def _gradient_v(img: Image.Image, top: tuple, bottom: tuple):
    """纵向渐变背景（逐行绘制）"""
    draw = ImageDraw.Draw(img)
    for y in range(VIDEO_H):
        t = y / VIDEO_H
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        draw.line([(0, y), (VIDEO_W, y)], fill=(r, g, b))


def _make_silence(duration: float, tag: str, temp_files: list) -> str:
    """生成指定时长的静音 mp3，返回文件路径"""
    path = os.path.join(OUTPUT_DIR, f"_sil_{tag}.mp3")
    ffmpeg_run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", f"{duration:.3f}", "-c:a", "libmp3lame", "-q:a", "9", path,
    ], f"静音_{tag}")
    temp_files.append(path)
    return path


# ===== 卡片生成器 =====

def generate_title_card(topic: str, pro_position: str, con_position: str) -> str:
    """标题卡：渐变背景 + 大标题 + 正反方对比 + 底部色条"""
    img = Image.new("RGB", (VIDEO_W, VIDEO_H))
    _gradient_v(img, (10, 10, 35), (5, 5, 18))
    draw = ImageDraw.Draw(img)

    f_label  = _load_font(17)
    f_topic  = _load_font(46)
    f_side   = _load_font(22)
    f_pos    = _load_font(19)
    f_vs     = _load_font(64)

    # 顶部品牌标签
    label = "A I   辩 论 赛"
    lw = draw.textbbox((0, 0), label, font=f_label)[2]
    draw.text(((VIDEO_W - lw) // 2, 44), label, fill=(110, 110, 175), font=f_label)

    # 辩题（居中，最多3行）
    topic_lines = _wrap_text(topic, f_topic, VIDEO_W - 220, draw)
    ty = 100
    for line in topic_lines[:3]:
        lw = draw.textbbox((0, 0), line, font=f_topic)[2]
        draw.text(((VIDEO_W - lw) // 2, ty), line, fill=(238, 238, 255), font=f_topic)
        ty += 58

    # 分割线
    div_y = ty + 14
    draw.line([(140, div_y), (VIDEO_W - 140, div_y)], fill=(50, 50, 100), width=1)

    # VS 布局
    ys = div_y + 26
    col_w = VIDEO_W // 2 - 80
    pro_x = 72
    con_x = VIDEO_W // 2 + 60

    # VS 字样（居中，置于正反中间）
    vsw = draw.textbbox((0, 0), "VS", font=f_vs)[2]
    draw.text(((VIDEO_W - vsw) // 2, ys - 6), "VS", fill=(65, 65, 125), font=f_vs)

    # 正方
    draw.text((pro_x, ys), "正 方", fill=(79, 172, 254), font=f_side)
    pro_lines = _wrap_text(pro_position, f_pos, col_w, draw)
    yp = ys + 34
    for ln in pro_lines[:4]:
        draw.text((pro_x, yp), ln, fill=(155, 195, 240), font=f_pos)
        yp += 27

    # 反方
    draw.text((con_x, ys), "反 方", fill=(254, 100, 100), font=f_side)
    con_lines = _wrap_text(con_position, f_pos, col_w, draw)
    yc = ys + 34
    for ln in con_lines[:4]:
        draw.text((con_x, yc), ln, fill=(240, 155, 155), font=f_pos)
        yc += 27

    # 底部蓝→红渐变色条
    for x in range(VIDEO_W):
        t = x / VIDEO_W
        r = int(79  + (254 - 79)  * t)
        g = int(172 + (100 - 172) * t)
        b = int(254 + (100 - 254) * t)
        draw.line([(x, VIDEO_H - 6), (x, VIDEO_H)], fill=(r, g, b))

    path = os.path.join(OUTPUT_DIR, "_title_card.png")
    img.save(path)
    return path


def generate_phase_card(phase_name: str) -> str:
    """阶段过渡卡：深色渐变 + 装饰线 + 大号文字"""
    img = Image.new("RGB", (VIDEO_W, VIDEO_H))
    _gradient_v(img, (12, 10, 32), (6, 5, 18))
    draw = ImageDraw.Draw(img)

    f_phase = _load_font(62)
    f_sub   = _load_font(17)

    cx, cy = VIDEO_W // 2, VIDEO_H // 2

    # 上装饰线
    draw.line([(cx - 260, cy - 62), (cx + 260, cy - 62)], fill=(70, 52, 130), width=1)

    # 阶段名
    pw = draw.textbbox((0, 0), phase_name, font=f_phase)[2]
    draw.text(((VIDEO_W - pw) // 2, cy - 40), phase_name, fill=(192, 144, 255), font=f_phase)

    # 下装饰线
    draw.line([(cx - 260, cy + 48), (cx + 260, cy + 48)], fill=(70, 52, 130), width=1)

    # 三个装饰点
    for dx in (-20, 0, 20):
        draw.ellipse([(cx + dx - 3, cy + 62), (cx + dx + 3, cy + 68)], fill=(80, 58, 140))

    # 底部小字
    sub = "— AI 辩论赛 —"
    sw = draw.textbbox((0, 0), sub, font=f_sub)[2]
    draw.text(((VIDEO_W - sw) // 2, VIDEO_H - 46), sub, fill=(55, 55, 95), font=f_sub)

    path = os.path.join(OUTPUT_DIR, f"_phase_{phase_name}.png")
    img.save(path)
    return path


def generate_speaker_card(name: str, side: str, role: str,
                          text: str, index: int, phase: str = "") -> str:
    """辩手发言卡：渐变背景 + 左侧色条 + 徽章标签 + 阶段标注"""
    if side == "pro":
        bg_top, bg_bot = (8, 20, 54), (4, 10, 30)
        accent          = (79, 172, 254)
        tag_bg          = (18, 46, 96)
        side_label      = "正方"
    else:
        bg_top, bg_bot = (44, 8, 8), (22, 4, 4)
        accent          = (254, 100, 100)
        tag_bg          = (88, 20, 20)
        side_label      = "反方"

    img = Image.new("RGB", (VIDEO_W, VIDEO_H))
    _gradient_v(img, bg_top, bg_bot)
    draw = ImageDraw.Draw(img)

    BAR  = 7   # 左侧色条宽度
    left = BAR + 50
    rmar = 60  # 右边距

    f_name = _load_font(36)
    f_tag  = _load_font(17)
    f_body = _load_font(28)
    f_hint = _load_font(15)

    # 左侧色条
    draw.rectangle([0, 0, BAR, VIDEO_H], fill=accent)

    # --- 标题行 ---
    header_y = 36
    draw.text((left, header_y), name, fill=accent, font=f_name)

    # 徽章（辩手角色）
    badge = f"{side_label} · {role}"
    bbox  = draw.textbbox((0, 0), badge, font=f_tag)
    bw    = bbox[2] - bbox[0] + 20
    bh    = bbox[3] - bbox[1] + 12
    nw    = draw.textbbox((0, 0), name, font=f_name)[2]
    bx    = left + nw + 16
    by    = header_y + 8
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=5, fill=tag_bg)
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=5, outline=accent, width=1)
    draw.text((bx + 10, by + 6), badge, fill=accent, font=f_tag)

    # 分割线
    div_y = header_y + 58
    draw.line([(left, div_y), (VIDEO_W - rmar, div_y)], fill=(44, 44, 80), width=1)

    # --- 正文 ---
    body_y   = div_y + 20
    line_h   = 44
    avail_h  = VIDEO_H - body_y - 50
    max_lines = avail_h // line_h

    lines = _wrap_text(text, f_body, VIDEO_W - left - rmar, draw)
    for i, ln in enumerate(lines[:max_lines]):
        draw.text((left, body_y + i * line_h), ln, fill=(225, 225, 230), font=f_body)
    if len(lines) > max_lines:
        draw.text((left, body_y + max_lines * line_h), "…（发言已截断）",
                  fill=(100, 100, 140), font=f_hint)

    # --- 底部阶段标注 ---
    if phase:
        phase_label = PHASE_DISPLAY_NAMES.get(phase, phase)
        pl_text = f"▸ {phase_label}"
        plw = draw.textbbox((0, 0), pl_text, font=f_hint)[2]
        draw.text((VIDEO_W - rmar - plw, VIDEO_H - 30),
                  pl_text, fill=(70, 70, 110), font=f_hint)

    path = os.path.join(OUTPUT_DIR, f"_card_{index:03d}.png")
    img.save(path)
    return path


def generate_judge_card(text: str) -> str:
    """裁判评判卡：绿色主题 + 标题 + Markdown 简单渲染"""
    img = Image.new("RGB", (VIDEO_W, VIDEO_H))
    _gradient_v(img, (8, 24, 8), (4, 12, 4))
    draw = ImageDraw.Draw(img)

    BAR  = 7
    left = BAR + 50
    rmar = 60

    f_title = _load_font(40)
    f_body  = _load_font(22)
    f_sub   = _load_font(18)
    f_hint  = _load_font(15)

    # 左侧绿色条
    draw.rectangle([0, 0, BAR, VIDEO_H], fill=(106, 254, 106))

    # 标题
    draw.text((left, 34), "裁判评判", fill=(106, 254, 106), font=f_title)
    draw.line([(left, 86), (VIDEO_W - rmar, 86)], fill=(28, 78, 28), width=1)

    # 简单清理 Markdown
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', text)      # 去掉加粗符号
    clean = re.sub(r'^## (.+)', r'▍ \1', clean, flags=re.MULTILINE)  # ## 转段落标题
    clean = re.sub(r'^\| .+', '', clean, flags=re.MULTILINE)          # 跳过表格行
    clean = re.sub(r'\n{3,}', '\n\n', clean).strip()

    lines = _wrap_text(clean, f_body, VIDEO_W - left - rmar, draw)
    y = 104
    line_h = 31
    max_lines = (VIDEO_H - y - 28) // line_h

    for ln in lines[:max_lines]:
        if ln.startswith("▍"):
            # 小节标题：用浅绿色高亮
            draw.text((left, y), ln[2:], fill=(130, 220, 130), font=f_sub)
        else:
            draw.text((left, y), ln, fill=(198, 218, 198), font=f_body)
        y += line_h

    if len(lines) > max_lines:
        draw.text((left, y), "…（点评已截断）", fill=(75, 135, 75), font=f_hint)

    path = os.path.join(OUTPUT_DIR, "_judge_card.png")
    img.save(path)
    return path


# ===== 主导出函数 =====

def export_debate_video(history: list[dict], topic: str,
                        pro_position: str = "", con_position: str = "") -> str:
    """
    将辩论记录导出为完整视频（纯文字卡片，无字幕）。

    Args:
        history: DebateEngine.history 列表
        topic: 辩题
        pro_position: 正方立场
        con_position: 反方立场

    Returns:
        输出视频文件路径
    """
    print("=" * 50)
    print("  开始导出辩论视频")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temp_files: list[str] = []
    audio_dir = os.path.join(OUTPUT_DIR, "audio")

    # 预生成可复用的静音文件
    sil_title  = _make_silence(TITLE_DURATION,      "title",  temp_files)
    sil_phase  = _make_silence(PHASE_CARD_DURATION,  "phase",  temp_files)
    sil_gap    = _make_silence(SPEECH_GAP,           "gap",    temp_files)

    # timeline_imgs  : [(img_path, duration), ...]   →  用于构建图片序列视频
    # audio_segments : [audio_file_path, ...]         →  逐段拼接成完整音轨
    # 两者逐段对应，保证音视频严格同步
    timeline_imgs:  list[tuple[str, float]] = []
    audio_segments: list[str]               = []

    # ---------- 标题卡 ----------
    title_img = generate_title_card(topic, pro_position, con_position)
    temp_files.append(title_img)
    timeline_imgs.append((title_img, TITLE_DURATION))
    audio_segments.append(sil_title)

    # ---------- 辩论过程 ----------
    current_phase_display = None
    card_index = 0

    for entry in history:
        if entry["id"] == "judge":
            continue

        phase_display = PHASE_DISPLAY_NAMES.get(entry["phase"], entry["phase"])

        # 阶段卡（显示名变化时插入）
        if phase_display != current_phase_display:
            current_phase_display = phase_display
            phase_img = generate_phase_card(phase_display)
            temp_files.append(phase_img)
            timeline_imgs.append((phase_img, PHASE_CARD_DURATION))
            audio_segments.append(sil_phase)

        # 辩手发言卡
        speaker_img = generate_speaker_card(
            entry["name"], entry["side"], entry["role"],
            entry["content"], card_index, entry.get("phase", ""),
        )
        temp_files.append(speaker_img)

        audio_file = os.path.join(audio_dir, f"turn_{card_index:03d}_{entry['id']}.mp3")
        if os.path.exists(audio_file):
            audio_dur = get_audio_duration(audio_file)
            card_dur  = audio_dur + SPEECH_GAP
            timeline_imgs.append((speaker_img, card_dur))
            audio_segments.append(audio_file)
            audio_segments.append(sil_gap)   # 发言后的静音间隔
        else:
            # 无对应音频（超时跳过等情况）：用静音填充
            fallback_dur = 3.0
            sil_fb = _make_silence(fallback_dur, f"fb_{card_index}", temp_files)
            timeline_imgs.append((speaker_img, fallback_dur))
            audio_segments.append(sil_fb)

        card_index += 1

    # ---------- 裁判点评卡 ----------
    judge_entries = [e for e in history if e["id"] == "judge"]
    if judge_entries:
        judge_img = generate_judge_card(judge_entries[0]["content"])
        temp_files.append(judge_img)

        # 阶段卡
        if current_phase_display != "裁判点评":
            phase_img = generate_phase_card("裁判点评")
            temp_files.append(phase_img)
            timeline_imgs.append((phase_img, PHASE_CARD_DURATION))
            audio_segments.append(sil_phase)

        judge_audio = os.path.join(audio_dir, f"turn_{card_index:03d}_judge.mp3")
        if os.path.exists(judge_audio):
            judge_dur = get_audio_duration(judge_audio)
            card_dur  = judge_dur + SPEECH_GAP
            timeline_imgs.append((judge_img, card_dur))
            audio_segments.append(judge_audio)
            audio_segments.append(sil_gap)
        else:
            sil_judge = _make_silence(10.0, "judge_fb", temp_files)
            timeline_imgs.append((judge_img, 10.0))
            audio_segments.append(sil_judge)

    # ===== [1/3] 图片序列 → 无声视频 =====
    print(f"  [1/3] 渲染卡片序列（共 {len(timeline_imgs)} 张）...")
    img_list = os.path.join(OUTPUT_DIR, "_img_list.txt")
    temp_files.append(img_list)
    with open(img_list, "w", encoding="utf-8") as f:
        for img_path, duration in timeline_imgs:
            f.write(f"file '{img_path.replace(chr(92), '/')}'\n")
            f.write(f"duration {duration:.3f}\n")
        # 最后一帧重复一次（ffmpeg concat 规范要求）
        f.write(f"file '{timeline_imgs[-1][0].replace(chr(92), '/')}'\n")

    temp_video = os.path.join(OUTPUT_DIR, "_temp_video.mp4")
    temp_files.append(temp_video)
    ffmpeg_run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", img_list,
        "-vf", (f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
                f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2:color=black"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", temp_video,
    ], "图片转视频")

    # ===== [2/3] 拼接音频（与视频严格同步）=====
    print(f"  [2/3] 拼接音频（共 {len(audio_segments)} 段）...")
    audio_concat = os.path.join(OUTPUT_DIR, "_audio_concat.txt")
    temp_files.append(audio_concat)
    with open(audio_concat, "w", encoding="utf-8") as f:
        for ap in audio_segments:
            f.write(f"file '{ap.replace(chr(92), '/')}'\n")

    merged_audio = os.path.join(OUTPUT_DIR, "_merged_audio.mp3")
    temp_files.append(merged_audio)
    ffmpeg_run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", audio_concat, "-c:a", "libmp3lame", "-q:a", "4", merged_audio,
    ], "拼接音频")

    # ===== [3/3] 合成最终视频 =====
    print("  [3/3] 合成最终视频...")
    output_path = os.path.join(OUTPUT_DIR, "debate_video.mp4")
    ffmpeg_run([
        "ffmpeg", "-y", "-i", temp_video, "-i", merged_audio,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path,
    ], "合成视频")

    # 清理临时文件
    for f in temp_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    print(f"\n  ✓ 视频已导出: {output_path}")
    return output_path
