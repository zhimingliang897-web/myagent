from flask import render_template, send_from_directory
import os

def register_frontend_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/login')
    def login_page():
        return render_template('login.html')
    
    @app.route('/register')
    def register_page():
        return render_template('register.html')
    
    @app.route('/debate')
    def debate_page():
        return render_template('debate.html')
    
    @app.route('/tokens')
    def tokens_page():
        return render_template('tokens.html')
    
    @app.route('/lines')
    def lines_page():
        return render_template('lines.html')
    
    @app.route('/compare')
    def compare_page():
        return render_template('compare.html')
    
    @app.route('/pdf')
    def pdf_page():
        return render_template('pdf.html')
    
    @app.route('/assets/<path:filename>')
    def serve_assets(filename):
        return send_from_directory(os.path.join(app.static_folder, 'assets'), filename)
    
    @app.route('/demos/<path:filename>')
    def serve_demos(filename):
        return send_from_directory(os.path.join(app.static_folder, 'demos'), filename)
    
    @app.route('/api/health')
    def health():
        return {'status': 'ok', 'service': 'Integrity Tools API', 'version': '1.0.0'}