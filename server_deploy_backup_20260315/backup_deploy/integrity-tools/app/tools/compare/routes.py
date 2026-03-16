from flask import Blueprint, request, jsonify
import asyncio
import aiohttp

from app.auth.routes import token_required
from config import LLM_PROVIDERS

compare_bp = Blueprint('compare', __name__)

async def call_provider_async(session, provider_key, provider, question, system_prompt, temperature):
    start_time = asyncio.get_event_loop().time()
    
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': question})
    
    try:
        async with session.post(
            f"{provider['base_url']}/chat/completions",
            headers={
                'Authorization': f"Bearer {provider['api_key']}",
                'Content-Type': 'application/json'
            },
            json={
                'model': provider['model'],
                'messages': messages,
                'temperature': temperature,
                'stream': False
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            if not resp.ok:
                text = await resp.text()
                return {'name': provider_key, 'error': f'请求失败 {resp.status}: {text[:100]}', 'latency_ms': latency_ms}
            
            data = await resp.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if not content:
                return {'name': provider_key, 'error': '未获取到有效内容', 'latency_ms': latency_ms}
            
            return {
                'name': provider_key,
                'model': provider['model'],
                'content': content,
                'latency_ms': latency_ms
            }
    except asyncio.TimeoutError:
        return {'name': provider_key, 'error': '请求超时', 'latency_ms': 0}
    except Exception as e:
        return {'name': provider_key, 'error': str(e)[:100], 'latency_ms': 0}

@compare_bp.route('/providers')
def get_providers():
    providers = []
    for key, prov in LLM_PROVIDERS.items():
        providers.append({
            'name': key,
            'models': [prov['model']],
            'baseUrl': prov['base_url'],
            'keyPresent': bool(prov.get('api_key'))
        })
    return jsonify({'providers': providers})

@compare_bp.route('', methods=['POST'])
@token_required
def compare():
    data = request.json
    question = data.get('question', '').strip()
    system_prompt = data.get('systemPrompt', '').strip()
    temperature = data.get('temperature', 0.7)
    requests_list = data.get('requests', [])
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    if not requests_list:
        return jsonify({'error': '至少选择一个模型'}), 400
    
    async def run_comparisons():
        async with aiohttp.ClientSession() as session:
            tasks = []
            for req in requests_list:
                provider_key = req.get('name', '')
                model = req.get('model', '')
                base_url = req.get('baseUrl', '')
                
                provider = LLM_PROVIDERS.get(provider_key, {})
                if provider:
                    api_key = provider.get('api_key', '')
                    model = model or provider.get('model', '')
                    base_url = base_url or provider.get('base_url', '')
                else:
                    api_key = ''
                
                if not api_key:
                    tasks.append(asyncio.create_task(
                        asyncio.sleep(0, result={'name': provider_key, 'error': '缺少API Key', 'latency_ms': 0})
                    ))
                    continue
                
                tasks.append(call_provider_async(
                    session, provider_key,
                    {'api_key': api_key, 'model': model, 'base_url': base_url},
                    question, system_prompt, temperature
                ))
            
            return await asyncio.gather(*tasks)
    
    results = asyncio.run(run_comparisons())
    return jsonify({'results': results})