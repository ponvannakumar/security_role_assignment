# ml/feature_extractor.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import re

class FeatureExtractor:
    """Feature extraction for security findings"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        self.security_patterns = {
            'aws_services': r'\b(?:s3|ec2|iam|vpc|rds|lambda)\b',
            'azure_services': r'\b(?:azure|blob|vm|ad|keyvault)\b',
            'gcp_services': r'\b(?:gcp|gce|gcs|bigquery)\b',
            'vulnerability_terms': r'\b(?:cve|vulnerability|exploit|patch)\b',
            'network_terms': r'\b(?:firewall|port|protocol|tcp|udp)\b',
            'access_terms': r'\b(?:permission|privilege|access|auth)\b'
        }
    
    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features"""
        return self.tfidf_vectorizer.fit_transform(texts).toarray()
    
    def extract_pattern_features(self, text: str) -> Dict[str, int]:
        """Extract pattern-based features"""
        features = {}
        
        for pattern_name, pattern in self.security_patterns.items():
            matches = len(re.findall(pattern, text.lower()))
            features[f'{pattern_name}_count'] = matches
            features[f'{pattern_name}_present'] = 1 if matches > 0 else 0
        
        return features
    
    def extract_text_stats(self, text: str) -> Dict[str, float]:
        """Extract text statistical features"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0
        }
    
    def extract_security_indicators(self, text: str) -> Dict[str, int]:
        """Extract security-specific indicators"""
        indicators = {
            'has_cve': 1 if re.search(r'cve-\d{4}-\d+', text.lower()) else 0,
            'has_ip_address': 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text) else 0,
            'has_domain': 1 if re.search(r'\b[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}\b', text) else 0,
            'has_port': 1 if re.search(r':\d{1,5}\b', text) else 0,
            'urgency_level': len(re.findall(r'\b(?:urgent|critical|immediate|emergency)\b', text.lower()))
        }
        
        return indicators
    
    def extract_all_features(self, text: str) -> Dict[str, any]:
        """Extract all features for a single text"""
        features = {}
        
        # Pattern features
        features.update(self.extract_pattern_features(text))
        
        # Text statistics
        features.update(self.extract_text_stats(text))
        
        # Security indicators
        features.update(self.extract_security_indicators(text))
        
        return features
    
    def fit_transform(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform texts to feature matrix"""
        # TF-IDF features
        tfidf_features = self.extract_tfidf_features(texts)
        
        # Additional features
        additional_features = []
        feature_names = []
        
        for text in texts:
            features = self.extract_all_features(text)
            if not feature_names:
                feature_names = list(features.keys())
            additional_features.append(list(features.values()))
        
        additional_features = np.array(additional_features)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, additional_features])
        
        # Feature names
        tfidf_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        all_feature_names = tfidf_names + feature_names
        
        return combined_features, all_feature_names