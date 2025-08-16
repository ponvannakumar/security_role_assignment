# config.py - Configuration Settings
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DATA_DIR = 'data'
    MODEL_DIR = 'models'
    LOG_DIR = 'logs'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # ML Model Settings
    MODEL_VERSION = '1.0'
    RETRAIN_THRESHOLD = 100
    
    # Advanced Features
    USE_BERT = True
    USE_ENSEMBLE = True
    ENABLE_LOGGING = True
    
    # API Settings
    API_RATE_LIMIT = '100/hour'
    ENABLE_CORS = True