"""Script to evaluate the ML model"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import SecurityRolePredictor
from ml.evaluator import ModelEvaluator
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    predictor = SecurityRolePredictor()
    evaluator = ModelEvaluator()
    
    # Load test data and evaluate
    logger.info("Evaluating model performance...")
    
    # Add evaluation logic here
    logger.info("Evaluation completed!")

if __name__ == '__main__':
    main()
