"""Deployment script"""

import subprocess
import sys
import os

def deploy():
    """Deploy the application"""
    print("🚀 Starting deployment...")
    
    # Create directories
    directories = ['data', 'models', 'logs', 'static/css', 'static/js', 'templates']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Download NLTK data
    print("📚 Downloading NLTK data...")
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    print("✅ Deployment completed successfully!")
    print("🌐 Run 'python app.py' to start the application")

if __name__ == '__main__':
    deploy()
