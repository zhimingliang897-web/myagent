from flask import Blueprint, request, jsonify
import requests
import time

from app.auth.routes import token_required
from config import LLM_PROVIDERS

tokens_bp = Blueprint('tokens', __name__)

TOKEN_PRICES = {
    'qwen-plus': {'input': 0.004, 'output': 0.012},
    'qwen-turbo': {'input': 0.001, 'output': 0.002},
    'qwen-max': {'input': 0.02, 'output': 0.06},
    'qwen-vl-plus': {'input': 0.008, 'output': 0.008},
    'gpt-4o': {'input': 0.0025, 'output': 0.01},
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
    'deepseek-chat': {'input': 0.0001, 'output': 0.0002},
    'doubao-1.5-pro-32k-250115': {'input': 0.0005, 'output': 0.001},
    'kimi-k2-turbo-preview': {'input': 0.001, 'output': 0.002},
}

TOKEN_RATIOS = {'zh': 0.5, 'en': 0.25}

@tokens_bp.route('/calc', methods=['POST'])
def calc_tokens():
    data = request.json
    model = data.get('model', 'qwen-plus')
    lang = data.get('lang', 'zh')
    chars = int(data.get('chars', 100))
    
    ratio = TOKEN_RATIOS.get(lang, 0.5)
    prompt_tokens = int(chars * ratio)
    completion_tokens = int(prompt_tokens * 0.3)
    prices = TOKEN_PRICES.get(model, TOKEN_PRICES['qwen-plus'])
    
    input_cost = (prompt_tokens / 1000) * prices['input']
    output_cost = (completion_tokens / 1000) * prices['output']
    total_cost = input_cost + output_cost
    
    return jsonify({
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
        'input_cost': round(input_cost, 6),
        'output_cost': round(output_cost, 6),
        'total_cost': round(total_cost, 6)
    })

@tokens_bp.route('/providers')
def get_providers():
    providers = []
    for key, prov in LLM_PROVIDERS.items():
        if prov.get('api_key'):
            providers.append({
                'id': key,
                'name': key.upper(),
                'model': prov['model'],
                'base_url': prov['base_url'],
                'has_key': True
            })
    return jsonify({'providers': providers})

@tokens_bp.route('/compare', methods=['POST'])
@token_required
def compare_models():
    data = request.json
    question = data.get('question', '').strip()
    models = data.get('models', [])
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    if not models:
        return jsonify({'error': '请选择至少一个模型'}), 400
    
    results = []
    for model_id in models:
        provider = LLM_PROVIDERS.get(model_id)
        if not provider or not provider.get('api_key'):
            results.append({
                'model': model_id,
                'error': 'API Key未配置'
            })
            continue
        
        start_time = time.time()
        try:
            resp = requests.post(
                f"{provider['base_url']}/chat/completions",
                headers={
                    'Authorization': f"Bearer {provider['api_key']}",
                    'Content-Type': 'application/json'
                },
                json={
                    'model': provider['model'],
                    'messages': [{'role': 'user', 'content': question}],
                    'temperature': 0.7
                },
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            
            latency = int((time.time() - start_time) * 1000)
            usage = data.get('usage', {})
            content = data['choices'][0]['message']['content']
            
            prices = TOKEN_PRICES.get(model_id, TOKEN_PRICES['qwen-plus'])
            cost = (usage.get('total_tokens', 0) / 1000) * (prices['input'] + prices['output'])
            
            results.append({
                'model': model_id,
                'model_name': provider['model'],
                'content': content,
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'latency_ms': latency,
                'cost': round(cost, 6)
            })
        except Exception as e:
            results.append({
                'model': model_id,
                'error': str(e)[:100]
            })
    
    return jsonify({'results': results})