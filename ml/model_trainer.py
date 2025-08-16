# ml/model_trainer.py
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from typing import Tuple, Dict, Any
import numpy as np

class ModelTrainer:
    """Model training utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.best_model = None
        
    def create_base_models(self) -> Dict[str, Any]:
        """Create base models for ensemble"""
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                class_weight='balanced'
            ),
            'lr': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'svm': SVC(
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        return models
    
    def create_ensemble_model(self) -> VotingClassifier:
        """Create ensemble model"""
        base_models = self.create_base_models()
        
        ensemble = VotingClassifier(
            estimators=list(base_models.items()),
            voting='soft'
        )
        
        return ensemble
    
    def train_model(self, X, y, model_type='ensemble') -> Any:
        """Train a single model"""
        if model_type == 'ensemble':
            model = self.create_ensemble_model()
        else:
            models = self.create_base_models()
            model = models.get(model_type, models['rf'])
        
        # Train model
        model.fit(X, y)
        
        return model
    
    def hyperparameter_tuning(self, X, y, model_type='rf') -> Any:
        """Perform hyperparameter tuning"""
        if model_type == 'rf':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'lr':
            model = LogisticRegression(random_state=42)
            param_grid = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        else:
            return self.train_model(X, y, model_type)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        self.logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def train_and_evaluate(self, X, y, test_size=0.2) -> Dict[str, Any]:
        """Train and evaluate multiple models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = {}
        
        # Train different models
        model_types = ['rf', 'lr', 'ensemble']
        
        for model_type in model_types:
            try:
                self.logger.info(f"Training {model_type} model...")
                
                if model_type in ['rf', 'lr']:
                    model = self.hyperparameter_tuning(X_train, y_train, model_type)
                else:
                    model = self.train_model(X_train, y_train, model_type)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[model_type] = {
                    'model': model,
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                self.logger.info(f"{model_type} accuracy: {accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {e}")
                continue
        
        # Select best model
        if results:
            best_model_type = max(results.keys(), key=lambda k: results[k]['accuracy'])
            self.best_model = results[best_model_type]['model']
            
            self.logger.info(f"Best model: {best_model_type} with accuracy: {results[best_model_type]['accuracy']:.3f}")
        
        return results
    
    def save_model(self, model, filepath: str):
        """Save trained model"""
        try:
            joblib.dump(model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            model = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None