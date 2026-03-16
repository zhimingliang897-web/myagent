from flask import Blueprint, request, jsonify, send_from_directory
import os
import re
import json
import threading
import shutil

from app.auth.routes import token_required
from app import db
from app.models import UserSession

lines_bp = Blueprint('lines', __name__)

LINES_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(LINES_DIR, exist_ok=True)

processing_status = {}

def get_day_dirs():
    if not os.path.exists(LINES_DIR):
        return []
    
    days = []
    for d in sorted(os.listdir(LINES_DIR)):
        if d.startswith('day') and os.path.isdir(os.path.join(LINES_DIR, d)):
            m = re.match(r'day(\d+)', d)
            if m:
                num = int(m.group(1))
                day_dir = os.path.join(LINES_DIR, d)
                txt = os.path.join(day_dir, f"{d}.txt")
                md = os.path.join(day_dir, f"{d}_study.md")
                
                word_count = 0
                if os.path.exists(txt):
                    with open(txt, 'r', encoding='utf-8') as f:
                        word_count = len([w for w in f.read().splitlines() if w.strip()])
                
                status = "empty"
                if os.path.exists(md):
                    status = "ready"
                elif os.path.exists(txt):
                    status = "pending"
                
                if num in processing_status:
                    ps = processing_status[num]
                    if ps["step"] not in ("done", "error"):
                        status = ps["step"]
                
                days.append({
                    "num": num,
                    "name": d,
                    "word_count": word_count,
                    "status": status,
                })
    return days

def parse_study_md(text: str) -> list:
    sections = re.split(r"\n## ", text)
    words = []
    
    for sec in sections:
        sec = sec.strip()
        if not sec or sec.startswith("#"):
            continue
        
        lines = sec.split("\n")
        word = lines[0].strip()
        
        meaning = ""
        meaning_match = re.search(r"\*\*meaning:\*\*\s*(.+?)(?=\n\n|\n\*\*|\Z)", sec, re.DOTALL)
        if meaning_match:
            meaning = meaning_match.group(1).strip()
        
        words.append({
            "word": word,
            "meaning": meaning,
        })
    
    return words

@lines_bp.route('/days')
def list_days():
    return jsonify(get_day_dirs())

@lines_bp.route('/day/<int:n>')
def get_day(n):
    day_name = f"day{n}"
    md_path = os.path.join(LINES_DIR, day_name, f"{day_name}_study.md")
    if not os.path.exists(md_path):
        return jsonify({"error": "Day not found"}), 404
    
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    words = parse_study_md(text)
    has_audio = os.path.exists(os.path.join(LINES_DIR, day_name, "audio"))
    
    return jsonify({"day": n, "words": words, "has_audio": has_audio})

@lines_bp.route('/upload', methods=['POST'])
@token_required
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    f = request.files['file']
    if not f.filename or not f.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files"}), 400
    
    day_num_str = request.form.get("day", "").strip()
    if not day_num_str or not day_num_str.isdigit() or int(day_num_str) < 1:
        return jsonify({"error": "Invalid day number"}), 400
    
    day_num = int(day_num_str)
    day_name = f"day{day_num}"
    day_dir = os.path.join(LINES_DIR, day_name)
    
    if os.path.exists(day_dir) and os.path.exists(os.path.join(day_dir, f"{day_name}.pdf")):
        return jsonify({"error": f"Day {day_num} already exists"}), 400
    
    os.makedirs(day_dir, exist_ok=True)
    pdf_path = os.path.join(day_dir, f"{day_name}.pdf")
    f.save(pdf_path)
    
    processing_status[day_num] = {"step": "pending", "error": None}
    
    return jsonify({"day": day_num, "status": "uploaded"})

@lines_bp.route('/status/<int:n>')
def status(n):
    if n in processing_status:
        return jsonify(processing_status[n])
    day_name = f"day{n}"
    if os.path.exists(os.path.join(LINES_DIR, day_name, f"{day_name}_study.md")):
        return jsonify({"step": "done", "error": None})
    return jsonify({"step": "unknown", "error": None})

@lines_bp.route('/day/<int:n>', methods=['DELETE'])
@token_required
def delete_day(n):
    day_name = f"day{n}"
    day_dir = os.path.join(LINES_DIR, day_name)
    if not os.path.exists(day_dir):
        return jsonify({"error": "Not found"}), 404
    shutil.rmtree(day_dir)
    processing_status.pop(n, None)
    return jsonify({"deleted": n})