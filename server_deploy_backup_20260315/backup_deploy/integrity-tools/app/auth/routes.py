from flask import Blueprint, request, jsonify, render_template
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import os
from functools import wraps

from app import db
from app.models import User
from config import INVITE_CODES, SECRET_KEY

auth_bp = Blueprint('auth', __name__)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': '缺少认证Token'}), 401
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            user = db.session.get(User, payload['user_id'])
            if not user:
                return jsonify({'error': '用户不存在'}), 401
            request.current_user = user
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token已过期'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': '无效Token'}), 401
        return f(*args, **kwargs)
    return decorated

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    invite_code = data.get('invite_code', '')
    
    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400
    if len(username) < 3 or len(username) > 20:
        return jsonify({'error': '用户名长度需在3-20字符之间'}), 400
    if len(password) < 6:
        return jsonify({'error': '密码长度至少6位'}), 400
    if invite_code not in INVITE_CODES:
        return jsonify({'error': '邀请码无效'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': '用户名已存在'}), 400
    
    user = User(
        username=username, 
        password_hash=generate_password_hash(password),
        invite_code=invite_code
    )
    db.session.add(user)
    db.session.commit()
    
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }, SECRET_KEY, algorithm='HS256')
    
    return jsonify({'message': '注册成功', 'token': token, 'username': username})

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': '用户名或密码错误'}), 401
    if not user.is_active:
        return jsonify({'error': '账户已被禁用'}), 403
    
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }, SECRET_KEY, algorithm='HS256')
    
    return jsonify({'message': '登录成功', 'token': token, 'username': username})

@auth_bp.route('/verify', methods=['GET'])
def verify():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({'valid': False}), 401
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        user = db.session.get(User, payload['user_id'])
        if not user:
            return jsonify({'valid': False}), 401
        return jsonify({'valid': True, 'username': user.username})
    except:
        return jsonify({'valid': False}), 401