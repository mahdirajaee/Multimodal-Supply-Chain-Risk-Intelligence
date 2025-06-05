#!/usr/bin/env python3

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.satellite_data import SatelliteRiskModel
from data.social_data import SentimentModel
from data.weather_data import WeatherModel
from data.economic_data import EconomicModel
from data.news_data import NewsModel
from fusion.risk_fusion import RiskFusionEngine, MultiModalTransformer

class TestModels(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 4
        
    def test_satellite_model(self):
        model = SatelliteRiskModel(num_classes=5)
        input_tensor = torch.randn(self.batch_size, 3, 224, 224)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 5))
        
    def test_sentiment_model(self):
        model = SentimentModel(num_classes=3)
        input_ids = torch.randint(0, 1000, (self.batch_size, 512))
        output = model(input_ids)
        self.assertEqual(output.shape, (self.batch_size, 3))
        
    def test_weather_model(self):
        model = WeatherModel(input_size=10, hidden_size=128)
        input_tensor = torch.randn(self.batch_size, 30, 10)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
    def test_economic_model(self):
        model = EconomicModel(input_size=15, d_model=512)
        input_tensor = torch.randn(self.batch_size, 60, 15)
        output = model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
    def test_news_model(self):
        model = NewsModel(num_classes=4)
        input_ids = torch.randint(0, 1000, (self.batch_size, 512))
        output = model(input_ids)
        self.assertEqual(output.shape, (self.batch_size, 4))

class TestFusionEngine(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'hidden_size': 512,
            'num_attention_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'learning_rate': 5e-5,
            'weight_decay': 0.01
        }
        self.fusion_engine = RiskFusionEngine(self.config)
        
    def test_multimodal_transformer(self):
        model = MultiModalTransformer()
        input_tensor = torch.randn(2, 1024)
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 1))
        
    def test_risk_fusion(self):
        predictions = {
            'satellite': torch.randn(1, 5),
            'sentiment': torch.randn(1, 3),
            'weather': torch.randn(1, 1),
            'economic': torch.randn(1, 1),
            'news': torch.randn(1, 4)
        }
        
        risk_scores = self.fusion_engine.fuse_predictions(predictions)
        
        self.assertIn('overall_risk', risk_scores)
        self.assertIn('operational_risk', risk_scores)
        self.assertIn('financial_risk', risk_scores)
        self.assertIn('reputational_risk', risk_scores)
        self.assertIn('confidence', risk_scores)
        
        for score in risk_scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

class TestDataLoaders(unittest.TestCase):
    
    def test_data_shapes(self):
        from data.satellite_data import SatelliteDataLoader
        from data.social_data import SocialDataLoader
        
        sat_config = {'cache_dir': '/tmp/test_satellite'}
        sat_loader = SatelliteDataLoader(sat_config)
        sat_data = sat_loader.get_processed_data()
        
        self.assertIn('images', sat_data)
        self.assertIn('labels', sat_data)
        
        social_config = {'cache_dir': '/tmp/test_social'}
        social_loader = SocialDataLoader(social_config)
        social_data = social_loader.get_processed_data()
        
        self.assertIn('input_ids', social_data)
        self.assertIn('labels', social_data)

if __name__ == '__main__':
    unittest.main()
