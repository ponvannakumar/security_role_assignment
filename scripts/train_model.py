# scripts/train_model.py
#!/usr/bin/env python3
"""Script to train the ML model"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import SecurityRolePredictor
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training...")
    
    predictor = SecurityRolePredictor()
    success = predictor.train_model()
    
    if success:
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()