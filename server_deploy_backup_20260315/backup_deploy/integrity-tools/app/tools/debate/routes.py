from flask import Blueprint, request, jsonify, Response
import json
import threading
import queue
import os
import time

from app.tools.debate.engine import DebateEngine
from app.tools.debate.tts import generate_turn_audio
from app.auth.routes import token_required
from config import DEBATERS, LLM_PROVIDERS, JUDGE, ZH_VOICES, MAX_WORDS, FREE_DEBATE_ROUNDS, DEBATE_TIME_LIMIT

import time

debate_bp = Blueprint('debate', __name__)

active_debate = None
debate_clients = []
debate_status = "idle"
debate_start_time = 0

@debate_bp.route('/config')
def get_config():
    providers_info = {}
    for key, prov in LLM_PROVIDERS.items():
        providers_info[key] = {
            "base_url": prov["base_url"],
            "model": prov["model"],
            "has_key": bool(prov.get("api_key", "").strip()),
        }
    
    debaters_info = []
    for d in DEBATERS:
        debaters_info.append({
            "id": d["id"],
            "name": d["name"],
            "side": d["side"],
            "role": d["role"],
            "provider": d["provider"],
            "voice": d["voice"],
            "personality": d["personality"],
        })
    
    return jsonify({
        "providers": providers_info,
        "debaters": debaters_info,
        "voices": ZH_VOICES,
        "params": {
            "max_words": MAX_WORDS,
            "free_debate_rounds": FREE_DEBATE_ROUNDS,
            "debate_time_limit": int(DEBATE_TIME_LIMIT),
        },
    })

@debate_bp.route('/start', methods=['POST'])
@token_required
def start_debate():
    global active_debate, debate_status, debate_start_time
    
    # 如果状态是 running 但已超时，自动重置
    if debate_status == "running":
        if active_debate and active_debate.finished:
            debate_status = "idle"
        elif time.time() - debate_start_time > DEBATE_TIME_LIMIT + 60:
            debate_status = "idle"
        else:
            return jsonify({"error": "辩论正在进行中，请等待或点击停止"}), 400
    
    data = request.json
    topic = data.get("topic", "").strip()
    pro_position = data.get("pro_position", "").strip()
    con_position = data.get("con_position", "").strip()
    
    if not topic or not pro_position or not con_position:
        return jsonify({"error": "请填写完整的辩题和正反方立场"}), 400
    
    custom_config = data.get("config", {})
    providers = LLM_PROVIDERS.copy()
    debaters = DEBATERS.copy()
    
    params = {
        "max_words": int(custom_config.get("params", {}).get("max_words", MAX_WORDS)),
        "free_debate_rounds": int(custom_config.get("params", {}).get("free_debate_rounds", FREE_DEBATE_ROUNDS)),
        "debate_time_limit": float(custom_config.get("params", {}).get("debate_time_limit", DEBATE_TIME_LIMIT)),
    }
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'audio')
    os.makedirs(output_dir, exist_ok=True)
    
    engine = DebateEngine(
        topic, pro_position, con_position,
        debaters=debaters, providers=providers, params=params,
        output_dir=output_dir
    )
    engine.init_clients()
    active_debate = engine
    debate_status = "running"
    debate_start_time = time.time()
    
    def run():
        global debate_status
        # 等待至少一个 SSE 客户端连接
        wait_count = 0
        while len(debate_clients) == 0 and wait_count < 50:
            time.sleep(0.1)
            wait_count += 1
        
        if len(debate_clients) == 0:
            print("[Debate] No SSE client connected, aborting")
            debate_status = "idle"
            return
        
        turn_index = 0
        try:
            for event in engine.run_debate():
                if event["event"] == "turn_end":
                    full_text = event["data"].get("full_text", "")
                    is_skipped = event["data"].get("skipped", False)
                    if full_text and not is_skipped:
                        try:
                            audio_info = generate_turn_audio(
                                event["data"]["speaker_id"],
                                full_text,
                                event["data"]["voice"],
                                turn_index,
                                output_dir
                            )
                            event["data"]["audio_url"] = f"/api/debate/audio/{os.path.basename(audio_info['audio_file'])}"
                            event["data"]["duration"] = audio_info["duration"]
                            turn_index += 1
                        except Exception as e:
                            print(f"[TTS Error] {e}")
                            event["data"]["audio_url"] = None
                    else:
                        event["data"]["audio_url"] = None
                
                for q in list(debate_clients):
                    q.put(event)
            
            debate_status = "finished"
        except Exception as e:
            print(f"[Debate Error] {e}")
            debate_status = "finished"
            for q in list(debate_clients):
                q.put({"event": "error", "data": {"message": str(e)}})
    
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})

@debate_bp.route('/stream')
def stream():
    q = queue.Queue()
    debate_clients.append(q)
    
    def event_stream():
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                except queue.Empty:
                    yield "event: ping\ndata: {}\n\n"
                    continue
                
                event_type = event.get("event", "message")
                data = json.dumps(event.get("data", {}), ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data}\n\n"
                
                if event_type in ("debate_end", "error"):
                    break
        finally:
            if q in debate_clients:
                debate_clients.remove(q)
    
    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@debate_bp.route('/stop', methods=['POST'])
def stop_debate():
    global active_debate, debate_status
    if active_debate:
        active_debate.stop()
        debate_status = "finished"
        return jsonify({"status": "stopped"})
    return jsonify({"status": "no active debate"})

@debate_bp.route('/status')
def status():
    return jsonify({"debate_status": debate_status})

@debate_bp.route('/audio/<filename>')
def serve_audio(filename):
    from flask import send_from_directory
    audio_dir = os.path.join(os.path.dirname(__file__), 'output', 'audio')
    return send_from_directory(audio_dir, filename)