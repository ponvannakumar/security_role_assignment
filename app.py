# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import os

# Download required NLTK data
for package in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        nltk.download(package)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityRolePredictor:
    def __init__(self):
        self.roles = [
            'Cloud Security Engineer', 'Infra Engineer', 'DevOps / Platform Engineer',
            'Cloud Admin', 'Vulnerability Analyst', 'App Owner', 'Sysadmin',
            'SOC Analyst', 'IAM Admin', 'Incident Responder', 'IR Team',
            'Security Compliance', 'Cloud Security Architect', 'IAM Specialist'
        ]
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            analyzer='word'
        )
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False

        self.role_keywords = {
            'Cloud Security Engineer': ['aws', 'azure', 'gcp', 'cloud', 'encryption', 'vpc', 'security group', 'firewall', 's3', 'iam policy'],
            'Infra Engineer': ['infrastructure', 'server', 'network', 'hardware', 'deployment', 'configuration', 'monitoring'],
            'DevOps / Platform Engineer': ['ci/cd', 'pipeline', 'deployment', 'automation', 'docker', 'kubernetes', 'jenkins', 'terraform'],
            'Cloud Admin': ['administration', 'provisioning', 'resource management', 'billing', 'account', 'subscription'],
            'Vulnerability Analyst': ['vulnerability', 'cve', 'patch', 'scan', 'assessment', 'security testing', 'penetration'],
            'App Owner': ['application', 'business logic', 'functionality', 'requirements', 'user access', 'data flow'],
            'Sysadmin': ['system administration', 'user management', 'permissions', 'maintenance', 'backup', 'recovery'],
            'SOC Analyst': ['monitoring', 'alerts', 'incidents', 'threat detection', 'log analysis', 'siem'],
            'IAM Admin': ['identity', 'access management', 'authentication', 'authorization', 'rbac', 'privileges'],
            'Incident Responder': ['incident response', 'investigation', 'forensics', 'containment', 'recovery'],
            'IR Team': ['incident response', 'team coordination', 'communication', 'escalation','ECR', 'immutability configured'],
            'Security Compliance': ['compliance', 'audit', 'governance', 'policy', 'regulations', 'standards'],
            'Cloud Security Architect': ['architecture', 'design', 'security framework', 'strategy', 'blueprint'],
            'IAM Specialist': ['identity management', 'access control', 'authentication systems', 'sso', 'federation']
        }

    def preprocess_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def extract_keywords(self, text, top_n=10):
        preprocessed = self.preprocess_text(text)
        try:
            tfidf_matrix = self.vectorizer.transform([preprocessed])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_indices = tfidf_scores.argsort()[-top_n:][::-1]
            return [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
        except:
            words = preprocessed.split()
            return [(word, 1.0) for word in set(words)[:top_n]]

    def calculate_keyword_similarity(self, finding_text):
        preprocessed_finding = self.preprocess_text(finding_text)
        finding_words = set(preprocessed_finding.split())
        similarities = {}
        for role, keywords in self.role_keywords.items():
            role_words = set()
            for keyword in keywords:
                role_words.update(self.preprocess_text(keyword).split())
            intersection = len(finding_words & role_words)
            union = len(finding_words | role_words)
            similarities[role] = intersection / union if union > 0 else 0
        return similarities

    def load_training_data(self):
        try:
            findings_df = pd.read_excel('data/findings_with_roles.xlsx', engine="openpyxl")
            findings_df.rename(columns=lambda x: x.strip().lower(), inplace=True)
            logger.info(f"Loaded {len(findings_df)} findings")

            if 'finding' not in findings_df.columns or 'role' not in findings_df.columns:
                logger.warning("Missing required columns in dataset. Falling back to sample data.")
                return self.create_sample_data()

            return findings_df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self.create_sample_data()

    def create_sample_data(self):
        sample_findings = [
            ("AWS S3 bucket has public read access enabled", "Cloud Security Engineer"),
            ("IAM user has excessive permissions", "IAM Admin"),
            ("Vulnerability CVE-2023-1234 found in application", "Vulnerability Analyst"),
            ("Security incident detected in network traffic", "SOC Analyst"),
            ("Cloud infrastructure misconfiguration", "Cloud Admin"),
            ("Application authentication bypass vulnerability", "App Owner"),
            ("System requires security patches", "Sysadmin"),
            ("Security compliance audit findings", "Security Compliance"),
            ("Identity federation configuration issue", "IAM Specialist"),
            ("Security architecture review needed", "Cloud Security Architect")
        ]
        return pd.DataFrame(sample_findings, columns=['finding', 'role'])

    def train_model(self):
        try:
            findings_df = self.load_training_data()

            X = findings_df['finding'].fillna('')
            y = findings_df['role'].fillna('')

            # Fix class imbalance issue (remove roles with only 1 sample OR disable stratify)
            role_counts = y.value_counts()
            if role_counts.min() < 2:
                logger.warning("Some roles have only 1 sample. Switching to non-stratified split.")
                stratify_option = None
            else:
                stratify_option = y

            X_processed = [self.preprocess_text(text) for text in X]
            X_vectorized = self.vectorizer.fit_transform(X_processed)

            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, test_size=0.2, random_state=42, stratify=stratify_option
            )

            self.model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, self.model.predict(X_test))
            logger.info(f"Model trained successfully. Accuracy: {accuracy:.3f}")
            self.save_model()
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict_role(self, finding_text):
        if not self.is_trained:
            if not self.load_model():
                if not self.train_model():
                    return {"error": "Model training failed"}

        try:
            processed_text = self.preprocess_text(finding_text)
            X_vectorized = self.vectorizer.transform([processed_text])
            probabilities = self.model.predict_proba(X_vectorized)[0]

            role_scores = {role: float(probabilities[i]) for i, role in enumerate(self.model.classes_)}
            keyword_similarities = self.calculate_keyword_similarity(finding_text)

            combined_scores = {role: 0.7 * role_scores.get(role, 0.0) + 0.3 * keyword_similarities.get(role, 0.0) for role in self.roles}
            sorted_roles = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

            top_role = sorted_roles[0][0]
            keywords = self.extract_keywords(finding_text)
            explanation = self.generate_explanation(finding_text, top_role, keywords, sorted_roles[:3])

            return {
                "predicted_role": top_role,
                "confidence_scores": dict(sorted_roles),
                "top_3_roles": sorted_roles[:3],
                "keywords_extracted": keywords,
                "explanation": explanation
            }
        except Exception as e:
            logger.error(f"Error predicting role: {e}")
            return {"error": str(e)}

    def generate_explanation(self, finding_text, predicted_role, keywords, top_roles):
        explanation = f"The finding was classified as '{predicted_role}' based on the following analysis:\n\n"
        explanation += "Key terms identified: " + ", ".join([kw[0] for kw in keywords[:5]]) + "\n\n"
        role_keywords = self.role_keywords.get(predicted_role, [])
        matching_keywords = [kw[0] for kw in keywords if any(kw[0] in rk.lower() or rk.lower() in kw[0] for rk in role_keywords)]
        if matching_keywords:
            explanation += f"These terms strongly associate with {predicted_role}: {', '.join(set(matching_keywords))}\n\n"
        explanation += "Confidence scores for top roles:\n"
        for role, score in top_roles:
            explanation += f"• {role}: {score:.2f}\n"
        return explanation

    def save_model(self):
        try:
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/role_classifier.pkl')
            joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self):
        try:
            if os.path.exists('models/role_classifier.pkl') and os.path.exists('models/vectorizer.pkl'):
                self.model = joblib.load('models/role_classifier.pkl')
                self.vectorizer = joblib.load('models/vectorizer.pkl')
                self.is_trained = True
                logger.info("Model loaded successfully")
                return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        return False

predictor = SecurityRolePredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        finding = data.get('finding', '').strip()
        if not finding:
            return jsonify({"error": "Please provide a finding to analyze"})
        result = predictor.predict_role(finding)
        logger.info(f"Prediction made for: {finding[:50]}... -> {result.get('predicted_role', 'Error')}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({"error": "An error occurred during prediction"})

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        success = predictor.train_model()
        return jsonify({"message": "Model retrained successfully"} if success else {"error": "Model retraining failed"})
    except Exception as e:
        logger.error(f"Error in retrain endpoint: {e}")
        return jsonify({"error": str(e)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_trained": predictor.is_trained})

if __name__ == '__main__':
    if not predictor.load_model():
        logger.info("No existing model found, training new model...")
        predictor.train_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
