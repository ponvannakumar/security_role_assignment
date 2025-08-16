# enhanced_model.py - Advanced ML Model with Additional Features
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import re
import joblib
import logging
from datetime import datetime
import json
import os

class AdvancedSecurityRolePredictor:
    def __init__(self):
        self.roles = [
            'Cloud Security Engineer', 'Infra Engineer', 'DevOps / Platform Engineer',
            'Cloud Admin', 'Vulnerability Analyst', 'App Owner', 'Sysadmin',
            'SOC Analyst', 'IAM Admin', 'Incident Responder', 'IR Team',
            'Security Compliance', 'Cloud Security Architect', 'IAM Specialist'
        ]
        
        # Multiple vectorizers for ensemble approach
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            analyzer='word',
            sublinear_tf=True
        )
        
        self.char_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(2, 5),
            analyzer='char',
            lowercase=True
        )
        
        # Ensemble model
        self.ensemble_model = None
        
        # BERT-based model for semantic understanding
        self.use_bert = True
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        except:
            self.use_bert = False
            logging.warning("BERT model not available, using traditional ML only")
        
        # Advanced feature extractors
        self.security_patterns = {
            'aws_services': r'\b(?:s3|ec2|iam|vpc|rds|lambda|cloudformation|cloudtrail)\b',
            'azure_services': r'\b(?:azure|blob|vm|ad|keyvault|resource.?group)\b',
            'gcp_services': r'\b(?:gcp|gce|gcs|bigquery|cloud.?function)\b',
            'vulnerability_terms': r'\b(?:cve|vulnerability|exploit|patch|security.?flaw)\b',
            'network_terms': r'\b(?:firewall|port|protocol|tcp|udp|dns|ssl|tls)\b',
            'access_terms': r'\b(?:permission|privilege|access|authentication|authorization)\b',
            'monitoring_terms': r'\b(?:log|alert|monitor|siem|incident|event)\b'
        }
        
        # Role priority matrix (for conflict resolution)
        self.role_priorities = {
            'critical_security': ['Security Compliance', 'Cloud Security Architect', 'SOC Analyst'],
            'immediate_response': ['Incident Responder', 'IR Team', 'SOC Analyst'],
            'technical_fix': ['Cloud Security Engineer', 'DevOps / Platform Engineer', 'Sysadmin'],
            'access_management': ['IAM Admin', 'IAM Specialist'],
            'vulnerability_handling': ['Vulnerability Analyst', 'Security Compliance']
        }
        
        self.logger = logging.getLogger(__name__)

    def extract_advanced_features(self, text):
        """Extract advanced features from text"""
        features = {}
        
        # Pattern-based features
        for pattern_name, pattern in self.security_patterns.items():
            matches = len(re.findall(pattern, text.lower()))
            features[f'{pattern_name}_count'] = matches
            features[f'{pattern_name}_present'] = 1 if matches > 0 else 0
        
        # Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Security severity indicators
        severity_terms = ['critical', 'high', 'medium', 'low', 'severe', 'urgent']
        features['severity_mentioned'] = any(term in text.lower() for term in severity_terms)
        
        # Urgency indicators
        urgency_terms = ['immediate', 'urgent', 'asap', 'emergency', 'critical']
        features['urgency_level'] = sum(1 for term in urgency_terms if term in text.lower())
        
        return features

    def get_bert_embeddings(self, texts):
        """Get BERT embeddings for texts"""
        if not self.use_bert:
            return None
        
        try:
            embeddings = []
            for text in texts:
                inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, 
                                           padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                    embeddings.append(embedding)
            return np.array(embeddings)
        except Exception as e:
            self.logger.warning(f"BERT embedding failed: {e}")
            return None

    def create_ensemble_model(self):
        """Create ensemble model with multiple algorithms"""
        models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=10, random_state=42, verbose=-1)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        
        self.ensemble_model = VotingClassifier(estimators=models, voting='soft')
        return self.ensemble_model

    def train_advanced_model(self, findings_df):
        """Train advanced ensemble model"""
        try:
            # Prepare text data
            X_text = findings_df['Finding'].fillna('').tolist()
            y = findings_df['Role'].fillna('').tolist()
            
            # TF-IDF features
            X_tfidf = self.tfidf_vectorizer.fit_transform(X_text)
            X_char = self.char_vectorizer.fit_transform(X_text)
            
            # Advanced features
            advanced_features = []
            for text in X_text:
                features = self.extract_advanced_features(text)
                advanced_features.append(list(features.values()))
            
            X_advanced = np.array(advanced_features)
            
            # BERT embeddings (if available)
            X_bert = self.get_bert_embeddings(X_text)
            
            # Combine all features
            feature_matrices = [X_tfidf.toarray(), X_char.toarray(), X_advanced]
            if X_bert is not None:
                feature_matrices.append(X_bert)
            
            X_combined = np.hstack(feature_matrices)
            
            # Create and train ensemble model
            self.create_ensemble_model()
            
            # Hyperparameter tuning for the ensemble
            param_grid = {
                'rf__n_estimators': [100, 200],
                'rf__max_depth': [15, 20, 25],
                'xgb__n_estimators': [50, 100],
                'xgb__max_depth': [8, 10, 12]
            }
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                self.ensemble_model, 
                param_grid, 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_combined, y)
            self.ensemble_model = grid_search.best_estimator_
            
            # Cross-validation score
            cv_scores = cross_val_score(self.ensemble_model, X_combined, y, cv=5)
            self.logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Feature importance analysis
            self.analyze_feature_importance(X_combined, y)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Advanced model training failed: {e}")
            return False

    def analyze_feature_importance(self, X, y):
        """Analyze and log feature importance"""
        try:
            # Train a simple random forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance = rf.feature_importances_
            
            # Log top 20 most important features
            top_indices = importance.argsort()[-20:][::-1]
            self.logger.info("Top 20 most important features:")
            for i, idx in enumerate(top_indices):
                self.logger.info(f"{i+1}. Feature {idx}: {importance[idx]:.4f}")
                
        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {e}")

    def predict_with_confidence_intervals(self, finding_text):
        """Predict with confidence intervals and uncertainty estimation"""
        try:
            # Prepare features (same as training)
            X_tfidf = self.tfidf_vectorizer.transform([finding_text])
            X_char = self.char_vectorizer.transform([finding_text])
            
            advanced_features = self.extract_advanced_features(finding_text)
            X_advanced = np.array([list(advanced_features.values())])
            
            X_bert = self.get_bert_embeddings([finding_text])
            
            # Combine features
            feature_matrices = [X_tfidf.toarray(), X_char.toarray(), X_advanced]
            if X_bert is not None:
                feature_matrices.append(X_bert)
            
            X_combined = np.hstack(feature_matrices)
            
            # Get predictions from individual models
            individual_predictions = {}
            for name, model in self.ensemble_model.named_estimators_.items():
                proba = model.predict_proba(X_combined)[0]
                individual_predictions[name] = dict(zip(self.ensemble_model.classes_, proba))
            
            # Get ensemble prediction
            ensemble_proba = self.ensemble_model.predict_proba(X_combined)[0]
            ensemble_pred = dict(zip(self.ensemble_model.classes_, ensemble_proba))
            
            # Calculate prediction uncertainty (standard deviation across models)
            uncertainties = {}
            for role in self.roles:
                if role in ensemble_pred:
                    individual_scores = [pred.get(role, 0) for pred in individual_predictions.values()]
                    uncertainties[role] = np.std(individual_scores)
            
            # Apply business rules for final decision
            final_prediction = self.apply_business_rules(finding_text, ensemble_pred)
            
            return {
                'ensemble_prediction': ensemble_pred,
                'individual_predictions': individual_predictions,
                'uncertainties': uncertainties,
                'final_prediction': final_prediction,
                'advanced_features': advanced_features
            }
            
        except Exception as e:
            self.logger.error(f"Advanced prediction failed: {e}")
            return None

    def apply_business_rules(self, finding_text, predictions):
        """Apply business rules to refine predictions"""
        text_lower = finding_text.lower()
        adjusted_predictions = predictions.copy()
        
        # Rule 1: IAM-related findings
        if any(term in text_lower for term in ['iam', 'identity', 'access', 'permission', 'privilege']):
            # Boost IAM-related roles
            for role in ['IAM Admin', 'IAM Specialist']:
                if role in adjusted_predictions:
                    adjusted_predictions[role] *= 1.2
        
        # Rule 2: Incident/Emergency situations
        if any(term in text_lower for term in ['incident', 'breach', 'attack', 'compromise']):
            # Boost incident response roles
            for role in ['Incident Responder', 'IR Team', 'SOC Analyst']:
                if role in adjusted_predictions:
                    adjusted_predictions[role] *= 1.15
        
        # Rule 3: Vulnerability findings
        if any(term in text_lower for term in ['vulnerability', 'cve', 'exploit', 'patch']):
            # Boost vulnerability analyst
            if 'Vulnerability Analyst' in adjusted_predictions:
                adjusted_predictions['Vulnerability Analyst'] *= 1.3
        
        # Rule 4: Cloud-specific terms
        cloud_terms = ['aws', 'azure', 'gcp', 'cloud', 's3', 'ec2', 'blob']
        if any(term in text_lower for term in cloud_terms):
            # Boost cloud-related roles
            for role in ['Cloud Security Engineer', 'Cloud Admin', 'Cloud Security Architect']:
                if role in adjusted_predictions:
                    adjusted_predictions[role] *= 1.1
        
        # Normalize adjusted predictions
        total = sum(adjusted_predictions.values())
        if total > 0:
            adjusted_predictions = {k: v/total for k, v in adjusted_predictions.items()}
        
        return adjusted_predictions

    def generate_detailed_report(self, finding_text, prediction_result):
        """Generate detailed analysis report"""
        if not prediction_result:
            return "Analysis failed"
        
        report = f"DETAILED SECURITY FINDING ANALYSIS REPORT\n"
        report += f"=" * 50 + "\n\n"
        report += f"Finding: {finding_text[:200]}...\n\n"
        
        # Top prediction
        final_pred = prediction_result['final_prediction']
        top_role = max(final_pred.items(), key=lambda x: x[1])
        report += f"RECOMMENDED ROLE: {top_role[0]} (Confidence: {top_role[1]:.1%})\n\n"
        
        # Confidence breakdown
        report += "CONFIDENCE BREAKDOWN:\n"
        sorted_roles = sorted(final_pred.items(), key=lambda x: x[1], reverse=True)
        for role, conf in sorted_roles[:5]:
            uncertainty = prediction_result['uncertainties'].get(role, 0)
            report += f"  {role}: {conf:.1%} (±{uncertainty:.2%})\n"
        
        # Feature analysis
        report += f"\nKEY FEATURES IDENTIFIED:\n"
        features = prediction_result['advanced_features']
        for feature, value in features.items():
            if value > 0:
                report += f"  {feature}: {value}\n"
        
        # Model agreement
        report += f"\nMODEL CONSENSUS:\n"
        individual_preds = prediction_result['individual_predictions']
        for model_name, preds in individual_preds.items():
            top_model_pred = max(preds.items(), key=lambda x: x[1])
            report += f"  {model_name.upper()}: {top_model_pred[0]} ({top_model_pred[1]:.1%})\n"
        
        return report

    def save_advanced_model(self, model_path='models/advanced_model'):
        """Save the advanced model"""
        try:
            os.makedirs(model_path, exist_ok=True)
            
            # Save main components
            joblib.dump(self.ensemble_model, f'{model_path}/ensemble_model.pkl')
            joblib.dump(self.tfidf_vectorizer, f'{model_path}/tfidf_vectorizer.pkl')
            joblib.dump(self.char_vectorizer, f'{model_path}/char_vectorizer.pkl')
            
            # Save configuration
            config = {
                'roles': self.roles,
                'security_patterns': self.security_patterns,
                'role_priorities': self.role_priorities,
                'use_bert': self.use_bert
            }
            
            with open(f'{model_path}/config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Advanced model saved to {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save advanced model: {e}")
            return False

    def load_advanced_model(self, model_path='models/advanced_model'):
        """Load the advanced model"""
        try:
            # Load main components
            self.ensemble_model = joblib.load(f'{model_path}/ensemble_model.pkl')
            self.tfidf_vectorizer = joblib.load(f'{model_path}/tfidf_vectorizer.pkl')
            self.char_vectorizer = joblib.load(f'{model_path}/char_vectorizer.pkl')
            
            # Load configuration
            with open(f'{model_path}/config.json', 'r') as f:
                config = json.load(f)
                self.roles = config['roles']
                self.security_patterns = config['security_patterns']
                self.role_priorities = config['role_priorities']
                self.use_bert = config.get('use_bert', False)
            
            self.logger.info(f"Advanced model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load advanced model: {e}")
            return False