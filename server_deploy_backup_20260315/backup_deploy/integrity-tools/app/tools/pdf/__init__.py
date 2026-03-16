from flask import Blueprint
pdf_bp = Blueprint('pdf', __name__)
from app.tools.pdf.routes import *