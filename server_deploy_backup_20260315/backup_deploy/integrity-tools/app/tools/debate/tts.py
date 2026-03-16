import asyncio
import os
import subprocess
import edge_tts

def generate_speech_sync(text: str, output_path: str, voice: str):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    async def _generate():
        communicate = edge_tts.Communicate(text, voice, rate="+0%")
        await communicate.save(output_path)
    
    loop.run_until_complete(_generate())

def get_audio_duration(file_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout:
            return float(result.stdout.decode("utf-8").strip())
    except Exception:
        pass
    return 0.0

def generate_turn_audio(speaker_id: str, text: str, voice: str, turn_index: int, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    audio_file = os.path.join(output_dir, f"turn_{turn_index:03d}_{speaker_id}.mp3")
    
    try:
        generate_speech_sync(text, audio_file, voice)
        duration = get_audio_duration(audio_file)
        return {"audio_file": audio_file, "duration": duration}
    except Exception as e:
        print(f"[TTS Error] {e}")
        return {"audio_file": None, "duration": 0.0}