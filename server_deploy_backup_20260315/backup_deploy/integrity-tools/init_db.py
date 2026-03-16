import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash
from config import INVITE_CODES

app = create_app()

with app.app_context():
    db.create_all()
    print("数据库初始化完成")
    print(f"有效邀请码: {INVITE_CODES}")