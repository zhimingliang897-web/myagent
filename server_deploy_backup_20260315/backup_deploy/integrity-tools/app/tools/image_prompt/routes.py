import os
import base64
import json
import uuid
from flask import Blueprint, request, jsonify, send_from_directory
from app.auth.routes import token_required
from app import db

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

image_prompt_bp = Blueprint('image_prompt', __name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'data', 'uploads')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'generated')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT_DALLE = """你是一个专业的 AI 绘画提示词工程师。请分析这张图片，生成一段适用于 DALL-E 3 / Midjourney 的英文提示词。

要求：
1. 使用连贯英文句子，不要堆砌碎片化关键词。
2. 描述主体外观、动作、表情与关键细节。
3. 描述光照、色彩、背景与整体风格。
4. 保持简洁精准，不要输出无关解释。
5. 只输出提示词正文。"""

SYSTEM_PROMPT_SD = """你是一个专业的 Stable Diffusion 提示词助手。请分析这张图片，生成适用于 SD WebUI / ComfyUI 的英文提示词（Tags）。

要求：
1. 使用英文 tags，并以逗号分隔。
2. 优先使用 Danbooru 风格标签。
3. 覆盖质量、风格、主体、背景、光影等信息。
4. 只输出 tags，不要输出解释。"""

MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def get_openai_client():
    if OpenAI is None:
        return None
    
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def analyze_image(image_data, style='dalle'):
    client = get_openai_client()
    if not client:
        return None, "API key not configured"
    
    system_prompt = SYSTEM_PROMPT_DALLE if style == 'dalle' else SYSTEM_PROMPT_SD
    
    try:
        response = client.chat.completions.create(
            model="qwen3-vl-flash-2026-01-22",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_data['mime_type']};base64,{image_data['base64']}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)


def generate_image(prompt, size='1024x1024', count=1, style='dalle'):
    client = get_openai_client()
    if not client:
        return None, "API key not configured"
    
    model = "dall-e-3"
    
    results = []
    try:
        for i in range(count):
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                n=1,
                quality="standard"
            )
            image_url = response.data[0].url
            results.append({
                "url": image_url,
                "revised_prompt": response.data[0].revised_prompt
            })
        return results, None
    except Exception as e:
        return None, str(e)


@image_prompt_bp.route('/analyze', methods=['POST'])
@token_required
def analyze(current_user):
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if not file.filename:
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    style = request.form.get('style', 'dalle')
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in MIME_BY_EXTENSION:
        return jsonify({'success': False, 'error': 'Unsupported file type'}), 400
    
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    file.save(temp_path)
    
    try:
        with open(temp_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        image_data = {
            'base64': image_b64,
            'mime_type': MIME_BY_EXTENSION[ext]
        }
        
        dalle_prompt, err = analyze_image(image_data, 'dalle')
        if err:
            return jsonify({'success': False, 'error': err}), 500
        
        sd_prompt, err = analyze_image(image_data, 'sd')
        if err:
            sd_prompt = ""
        
        return jsonify({
            'success': True,
            'prompt': dalle_prompt,
            'sd_prompt': sd_prompt
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@image_prompt_bp.route('/generate', methods=['POST'])
@token_required
def generate(current_user):
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'success': False, 'error': 'No prompt provided'}), 400
    
    size = data.get('size', '1024x1024')
    count = min(int(data.get('count', 1)), 4)
    
    results, err = generate_image(prompt, size, count)
    if err:
        return jsonify({'success': False, 'error': err}), 500
    
    return jsonify({
        'success': True,
        'images': results,
        'optimized_prompt': results[0].get('revised_prompt', prompt) if results else prompt
    })


@image_prompt_bp.route('/health', methods=['GET'])
def health():
    has_api_key = bool(os.environ.get('DASHSCOPE_API_KEY'))
    return jsonify({
        'status': 'ok' if has_api_key else 'missing_api_key',
        'has_api_key': has_api_key
    })