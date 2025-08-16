import unittest
from app import SecurityRolePredictor

class TestModel(unittest.TestCase):
    def setUp(self):
        self.predictor = SecurityRolePredictor()
    
    def test_preprocess_text(self):
        text = "AWS S3 bucket has public access!"
        processed = self.predictor.preprocess_text(text)
        self.assertIsInstance(processed, str)
        self.assertNotIn('!', processed)
    
    def test_extract_keywords(self):
        text = "IAM user has excessive permissions"
        keywords = self.predictor.extract_keywords(text)
        self.assertIsInstance(keywords, list)

if __name__ == '__main__':
    unittest.main()

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