# ml/evaluator.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import logging

class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self, model, X_test, y_test, class_names=None) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        # Per-class metrics
        if class_names is None:
            class_names = sorted(set(y_test))
        
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'precision': precision[i] if i < len(precision) else 0,
                'recall': recall[i] if i < len(recall) else 0,
                'f1_score': f1[i] if i < len(f1) else 0,
                'support': support[i] if i < len(support) else 0
            }
        
        # Overall metrics
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        results = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'class_names': class_names
        }
        
        if y_pred_proba is not None:
            results['prediction_probabilities'] = y_pred_proba
        
        return results
    
    def cross_validate_model(self, model, X, y, cv=5) -> Dict[str, float]:
        """Perform cross-validation"""
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_mean_accuracy': scores.mean(),
            'cv_std_accuracy': scores.std(),
            'cv_scores': scores.tolist()
        }
    
    def compare_models(self, models: Dict[str, Any], X_test, y_test) -> pd.DataFrame:
        """Compare multiple models"""
        results = []
        
        for model_name, model in models.items():
            try:
                evaluation = self.evaluate_model(model, X_test, y_test)
                
                results.append({
                    'Model': model_name,
                    'Accuracy': evaluation['accuracy'],
                    'Precision': evaluation['macro_precision'],
                    'Recall': evaluation['macro_recall'],
                    'F1-Score': evaluation['macro_f1']
                })
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def plot_confusion_matrix(self, confusion_matrix, class_names, title='Confusion Matrix'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        return plt
    
    def plot_feature_importance(self, model, feature_names, top_n=20):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            self.logger.warning("Model doesn't have feature importance")
            return None
        
        # Get top features
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        return plt
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate text evaluation report"""
        report = "MODEL EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Overall metrics
        report += f"Overall Accuracy: {evaluation_results['accuracy']:.3f}\n"
        report += f"Macro Precision: {evaluation_results['macro_precision']:.3f}\n"
        report += f"Macro Recall: {evaluation_results['macro_recall']:.3f}\n"
        report += f"Macro F1-Score: {evaluation_results['macro_f1']:.3f}\n\n"
        
        # Per-class metrics
        report += "PER-CLASS METRICS:\n"
        report += "-" * 20 + "\n"
        
        for class_name, metrics in evaluation_results['per_class_metrics'].items():
            report += f"\n{class_name}:\n"
            report += f"  Precision: {metrics['precision']:.3f}\n"
            report += f"  Recall: {metrics['recall']:.3f}\n"
            report += f"  F1-Score: {metrics['f1_score']:.3f}\n"
            report += f"  Support: {metrics['support']}\n"
        
        return report
    
    def calculate_prediction_confidence(self, probabilities) -> Dict[str, float]:
        """Calculate prediction confidence metrics"""
        max_probs = np.max(probabilities, axis=1)
        
        return {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs),
            'high_confidence_ratio': np.mean(max_probs > 0.8),
            'low_confidence_ratio': np.mean(max_probs < 0.5)
        }