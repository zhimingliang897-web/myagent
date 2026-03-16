"""Edge TTS 语音引擎 — 为辩论发言生成中文语音"""

import asyncio
import os
import subprocess
import edge_tts
from config import OUTPUT_DIR, SPEECH_GAP


async def generate_speech(text: str, output_path: str, voice: str, rate: str = "+0%"):
    """生成单段语音"""
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


def get_audio_duration(file_path: str) -> float:
    """用 ffprobe 获取音频时长（秒）"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True,
    )
    return float(result.stdout.decode("utf-8").strip())


def generate_speech_sync(text: str, output_path: str, voice: str):
    """同步版本的语音生成（在线程中调用）"""
    # 在 Windows 上，使用 asyncio.run() 在线程中会导致事件循环问题
    # 改用 get_event_loop() 或创建新的事件循环
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(generate_speech(text, output_path, voice))


def generate_turn_audio(speaker_id: str, text: str, voice: str, turn_index: int) -> dict:
    """
    为一段发言生成语音文件，返回音频信息。

    Returns:
        {"audio_file": str, "duration": float}
    """
    audio_dir = os.path.join(OUTPUT_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    audio_file = os.path.join(audio_dir, f"turn_{turn_index:03d}_{speaker_id}.mp3")
    generate_speech_sync(text, audio_file, voice)
    duration = get_audio_duration(audio_file)

    return {"audio_file": audio_file, "duration": duration}


def ffmpeg_run(args, desc=""):
    """运行 FFmpeg 命令"""
    result = subprocess.run(args, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg 失败 ({desc}): {stderr[-300:]}")
    return result


def merge_all_audio(audio_infos: list[dict], output_path: str) -> str:
    """
    将多段语音按顺序拼接为一个完整音频文件。

    Args:
        audio_infos: [{"audio_file": str, "duration": float}, ...]
        output_path: 输出文件路径

    Returns:
        输出文件路径
    """
    audio_dir = os.path.dirname(output_path)

    # 生成静音间隔
    silence = os.path.join(audio_dir, "_silence.mp3")
    ffmpeg_run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
        "-t", str(SPEECH_GAP), "-c:a", "libmp3lame", silence,
    ], "生成静音")

    # 写拼接列表
    concat_list = os.path.join(audio_dir, "_concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for info in audio_infos:
            f.write(f"file '{info['audio_file'].replace(chr(92), '/')}'\n")
            f.write(f"file '{silence.replace(chr(92), '/')}'\n")

    # 拼接
    ffmpeg_run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c:a", "libmp3lame", output_path,
    ], "拼接音频")

    # 清理
    os.remove(silence)
    os.remove(concat_list)

    return output_path
