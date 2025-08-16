"""Test suite for Security Role Assignment System"""

# tests/test_api.py
import unittest
import json
from app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
    
    def test_predict_endpoint(self):
        payload = {'finding': 'AWS S3 bucket has public access'}
        response = self.app.post('/predict', 
                               data=json.dumps(payload),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()