#!/usr/bin/env python3

import unittest
import requests
import json
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from api.server import app
from api.predictor import RiskPredictor
from fastapi.testclient import TestClient

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
        
    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("status", data)
        
    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertEqual(data["status"], "healthy")
        
    def test_model_status_endpoint(self):
        response = self.client.get("/models/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        
    @patch('api.server.predictor_instance')
    def test_predict_endpoint(self, mock_predictor):
        mock_predictor.predict_risk.return_value = {
            'timestamp': '2024-01-01T00:00:00',
            'timeframe_days': 30,
            'risk_scores': {
                'overall_risk': 0.5,
                'operational_risk': 0.6,
                'financial_risk': 0.4,
                'reputational_risk': 0.3,
                'confidence': 0.8
            },
            'individual_predictions': {
                'satellite': [0.1, 0.2, 0.3, 0.2, 0.2],
                'sentiment': [0.3, 0.4, 0.3],
                'weather': [0.5],
                'economic': [0.4],
                'news': [0.2, 0.3, 0.3, 0.2]
            },
            'recommendations': ['Monitor key suppliers closely']
        }
        
        request_data = {
            'timeframe_days': 30,
            'include_recommendations': True,
            'confidence_threshold': 0.7
        }
        
        response = self.client.post("/predict", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('risk_scores', data)
        self.assertIn('recommendations', data)
        self.assertIn('timestamp', data)

class TestPredictorLogic(unittest.TestCase):
    
    def setUp(self):
        self.mock_model_manager = MagicMock()
        self.mock_fusion_engine = MagicMock()
        self.predictor = RiskPredictor(self.mock_model_manager, self.mock_fusion_engine)
        
    def test_recommendation_generation(self):
        high_risk_scores = {
            'overall_risk': 0.9,
            'operational_risk': 0.95,
            'financial_risk': 0.8,
            'reputational_risk': 0.7,
            'confidence': 0.9
        }
        
        recommendations = self.predictor._generate_recommendations(high_risk_scores)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any('CRITICAL' in rec for rec in recommendations))
        
    def test_medium_risk_recommendations(self):
        medium_risk_scores = {
            'overall_risk': 0.5,
            'operational_risk': 0.6,
            'financial_risk': 0.4,
            'reputational_risk': 0.3,
            'confidence': 0.7
        }
        
        recommendations = self.predictor._generate_recommendations(medium_risk_scores)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any('MEDIUM' in rec for rec in recommendations))

if __name__ == '__main__':
    unittest.main()
