import torch
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from models.model_manager import ModelManager
from fusion.risk_fusion import RiskFusionEngine

logger = logging.getLogger(__name__)

class RiskPredictor:
    def __init__(self, model_manager: ModelManager, fusion_engine: RiskFusionEngine):
        self.model_manager = model_manager
        self.fusion_engine = fusion_engine
        
    def predict_risk(self, timeframe_days: int = 30) -> Dict[str, Any]:
        logger.info(f"Predicting risk for {timeframe_days} days ahead")
        
        predictions = {}
        models = self.model_manager.models
        
        for model_name, model in models.items():
            model.eval()
            with torch.no_grad():
                if model_name == 'satellite':
                    sample_input = torch.randn(1, 3, 224, 224)
                    pred = model(sample_input)
                elif model_name in ['sentiment', 'news']:
                    sample_input = torch.randint(0, 1000, (1, 512))
                    pred = model(sample_input)
                elif model_name in ['weather', 'economic']:
                    sample_input = torch.randn(1, 30, 10) if model_name == 'weather' else torch.randn(1, 60, 15)
                    pred = model(sample_input)
                
                predictions[model_name] = pred
        
        risk_scores = self.fusion_engine.fuse_predictions(predictions)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'timeframe_days': timeframe_days,
            'risk_scores': risk_scores,
            'individual_predictions': {k: v.mean().item() for k, v in predictions.items()},
            'recommendations': self._generate_recommendations(risk_scores)
        }
        
    def _generate_recommendations(self, risk_scores: Dict[str, float]) -> List[str]:
        recommendations = []
        overall_risk = risk_scores['overall_risk']
        
        if overall_risk > 0.8:
            recommendations.extend([
                "CRITICAL: Immediate supply chain diversification required",
                "Activate emergency supplier protocols",
                "Increase inventory buffers for critical components"
            ])
        elif overall_risk > 0.6:
            recommendations.extend([
                "HIGH: Review supplier contracts and contingency plans",
                "Consider alternative sourcing options",
                "Monitor key suppliers closely"
            ])
        elif overall_risk > 0.4:
            recommendations.extend([
                "MEDIUM: Maintain current monitoring protocols",
                "Prepare contingency communications",
                "Review logistics routing options"
            ])
        else:
            recommendations.append("LOW: Continue standard monitoring procedures")
            
        return recommendations
