# api/__init__.py
"""
API module for Security Role Assignment System
"""

from flask import Blueprint

api_bp = Blueprint('api', __name__, url_prefix='/api')

from . import routes