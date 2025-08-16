# setup.py - Package Setup
from setuptools import setup, find_packages

setup(
    name="security_role_assignment",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "Flask>=2.3.3",
        "Flask-CORS>=4.0.0",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.1",
        "joblib>=1.3.2",
        "openpyxl>=3.1.2",
        "xgboost>=1.7.6",
        "lightgbm>=4.0.0",
    ],
    author="Security Team",
    author_email="security@company.com",
    description="AI-powered security findings role assignment system",
    python_requires=">=3.8",
)