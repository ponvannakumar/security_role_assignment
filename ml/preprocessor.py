# ml/preprocessor.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Union

class TextPreprocessor:
    """Text preprocessing utilities for security findings"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Security-specific terms to preserve
        self.security_terms = {
            'aws', 'azure', 'gcp', 's3', 'ec2', 'iam', 'vpc',
            'ssl', 'tls', 'ssh', 'rdp', 'smtp', 'http', 'https',
            'cve', 'vulnerability', 'exploit', 'malware',
            'firewall', 'intrusion', 'breach', 'incident'
        }
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep hyphens and underscores
        text = re.sub(r'[^\w\s\-_]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords while preserving security terms"""
        return [token for token in tokens 
                if token not in self.stop_words or token in self.security_terms]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Filter short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_entities(self, text: str) -> dict:
        """Extract security-related entities"""
        entities = {
            'cloud_providers': [],
            'services': [],
            'vulnerabilities': [],
            'protocols': []
        }
        
        # Cloud providers
        cloud_patterns = {
            'aws': r'\b(aws|amazon|ec2|s3|iam|vpc)\b',
            'azure': r'\b(azure|microsoft|blob|ad)\b',
            'gcp': r'\b(gcp|google|gce|gcs)\b'
        }
        
        for provider, pattern in cloud_patterns.items():
            if re.search(pattern, text.lower()):
                entities['cloud_providers'].append(provider)
        
        # Vulnerabilities
        cve_pattern = r'\bcve-\d{4}-\d+\b'
        cves = re.findall(cve_pattern, text.lower())
        entities['vulnerabilities'].extend(cves)
        
        return entities