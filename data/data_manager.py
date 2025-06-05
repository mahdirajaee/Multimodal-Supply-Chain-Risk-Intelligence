import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .satellite_data import SatelliteDataLoader
from .social_data import SocialDataLoader
from .weather_data import WeatherDataLoader
from .economic_data import EconomicDataLoader
from .news_data import NewsDataLoader

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loaders = {
            'satellite': SatelliteDataLoader(config.satellite),
            'social': SocialDataLoader(config.social),
            'weather': WeatherDataLoader(config.weather),
            'economic': EconomicDataLoader(config.economic),
            'news': NewsDataLoader(config.news)
        }
        
    def prepare_training_data(self) -> None:
        logger.info("Preparing training data from all sources")
        for name, loader in self.loaders.items():
            logger.info(f"Loading {name} data...")
            loader.load_historical_data()
            
    def get_training_data(self) -> Dict[str, Any]:
        return {name: loader.get_processed_data() for name, loader in self.loaders.items()}
        
    def get_real_time_data(self) -> Dict[str, Any]:
        logger.info("Fetching real-time data")
        return {name: loader.get_current_data() for name, loader in self.loaders.items()}
        
    def sync_data_sources(self) -> None:
        for loader in self.loaders.values():
            loader.sync_cache()
