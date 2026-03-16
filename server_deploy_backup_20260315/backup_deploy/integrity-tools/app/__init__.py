import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../data/app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    
    CORS(app, origins=[
        'https://zhimingliang897-web.github.io',
        'http://localhost:3000',
        'http://localhost:5000',
        'http://localhost:5500',
        'http://8.138.164.133:5000'
    ])
    
    db.init_app(app)
    
    from app.auth.routes import auth_bp
    from app.tools.debate import debate_bp
    from app.tools.tokens import tokens_bp
    from app.tools.lines import lines_bp
    from app.tools.compare import compare_bp
    from app.tools.pdf import pdf_bp
    from app.tools.image_prompt import image_prompt_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(debate_bp, url_prefix='/api/debate')
    app.register_blueprint(tokens_bp, url_prefix='/api/tokens')
    app.register_blueprint(lines_bp, url_prefix='/api/lines')
    app.register_blueprint(compare_bp, url_prefix='/api/compare')
    app.register_blueprint(pdf_bp, url_prefix='/api/pdf')
    app.register_blueprint(image_prompt_bp, url_prefix='/api/tools/image-prompt')
    
    from app.routes import register_frontend_routes
    register_frontend_routes(app)
    
    return app