import os
import logging
from pathlib import Path
from typing import Dict, Any
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from data.data_manager import DataManager
from models.model_manager import ModelManager
from fusion.risk_fusion import RiskFusionEngine
from api.predictor import RiskPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    logger.info("Starting Supply Chain Risk Intelligence System")
    
    data_manager = DataManager(cfg.data)
    model_manager = ModelManager(cfg.models)
    fusion_engine = RiskFusionEngine(cfg.fusion)
    predictor = RiskPredictor(model_manager, fusion_engine)
    
    if cfg.mode == "train":
        logger.info("Training models...")
        data_manager.prepare_training_data()
        model_manager.train_all_models(data_manager.get_training_data())
        
    elif cfg.mode == "predict":
        logger.info("Running predictions...")
        predictions = predictor.predict_risk(cfg.prediction.timeframe)
        logger.info(f"Risk predictions: {predictions}")
        
    elif cfg.mode == "api":
        logger.info("Starting API server...")
        from api.server import start_server
        start_server(predictor, cfg.api)

if __name__ == "__main__":
    main()
