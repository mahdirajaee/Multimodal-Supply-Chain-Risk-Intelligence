import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List
from pathlib import Path
from data.satellite_data import SatelliteRiskModel
from data.social_data import SentimentModel
from data.weather_data import WeatherModel
from data.economic_data import EconomicModel
from data.news_data import NewsModel

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self._initialize_models()
        
    def _initialize_models(self) -> None:
        self.models = {
            'satellite': SatelliteRiskModel(self.config.satellite_model.num_classes),
            'sentiment': SentimentModel(self.config.sentiment_model.num_classes),
            'weather': WeatherModel(
                hidden_size=self.config.weather_model.hidden_size,
                num_layers=self.config.weather_model.num_layers
            ),
            'economic': EconomicModel(
                d_model=self.config.economic_model.d_model,
                nhead=self.config.economic_model.nhead,
                num_layers=self.config.economic_model.num_layers
            ),
            'news': NewsModel(self.config.news_model.num_classes)
        }
        
    def train_all_models(self, training_data: Dict[str, Any]) -> None:
        logger.info("Training all models")
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} model")
            self._train_single_model(model_name, model, training_data[model_name])
            
    def _train_single_model(self, name: str, model: nn.Module, data: Dict[str, torch.Tensor]) -> None:
        config = getattr(self.config, f"{name}_model")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss() if 'num_classes' in config else nn.MSELoss()
        
        model.train()
        for epoch in range(10):
            if name == 'satellite':
                outputs = model(data['images'])
                loss = criterion(outputs, data['labels'])
            elif name in ['sentiment', 'news']:
                outputs = model(data['input_ids'])
                loss = criterion(outputs, data['labels'])
            elif name in ['weather', 'economic']:
                outputs = model(data['sequences'])
                loss = criterion(outputs, data['labels'])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        logger.info(f"Completed training {name} model")
        
    def get_model(self, name: str) -> nn.Module:
        return self.models.get(name)
        
    def save_models(self, checkpoint_dir: str) -> None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            torch.save(model.state_dict(), checkpoint_path / f"{name}_model.pth")
            
    def load_models(self, checkpoint_dir: str) -> None:
        checkpoint_path = Path(checkpoint_dir)
        
        for name, model in self.models.items():
            model_path = checkpoint_path / f"{name}_model.pth"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded {name} model from checkpoint")
