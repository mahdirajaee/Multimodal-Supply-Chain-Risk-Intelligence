#!/usr/bin/env python3
"""
Enhanced Risk Predictor with Caching, Performance Optimization, and Model Ensemble
Includes model caching, batch processing, confidence intervals, and ensemble methods
"""

import torch
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from models.model_manager import ModelManager
from fusion.risk_fusion import RiskFusionEngine

logger = logging.getLogger(__name__)

@dataclass
class PredictionMetrics:
    """Metrics for prediction performance tracking"""
    processing_time_ms: float
    model_inference_time_ms: float
    fusion_time_ms: float
    cache_hit: bool
    confidence_score: float
    model_agreement: float  # How much models agree
    prediction_uncertainty: float

@dataclass
class RiskPredictionResult:
    """Enhanced risk prediction result"""
    risk_scores: Dict[str, float]
    individual_predictions: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    model_contributions: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    recommendations: List[str]
    prediction_metrics: PredictionMetrics
    timestamp: datetime
    model_versions: Dict[str, str]

class ModelCache:
    """Intelligent model output caching system"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_key(self, inputs: Dict[str, Any]) -> str:
        """Generate cache key from inputs"""
        key_str = str(sorted(inputs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available and valid"""
        key = self._generate_key(inputs)
        
        if key in self.cache:
            cached_time, result = self.cache[key]
            
            # Check TTL
            if time.time() - cached_time < self.ttl_seconds:
                self.access_times[key] = time.time()
                return result
            else:
                # Remove expired entry
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    def put(self, inputs: Dict[str, Any], result: Dict[str, Any]):
        """Cache prediction result"""
        key = self._generate_key(inputs)
        current_time = time.time()
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = (current_time, result)
        self.access_times[key] = current_time
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        now = time.time()
        valid_entries = sum(1 for cached_time, _ in self.cache.values() 
                          if now - cached_time < self.ttl_seconds)
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }

class EnhancedRiskPredictor:
    """Enhanced risk predictor with caching, ensembling, and performance optimization"""
    
    def __init__(self, model_manager: ModelManager, fusion_engine: RiskFusionEngine, config: Dict[str, Any] = None):
        self.model_manager = model_manager
        self.fusion_engine = fusion_engine
        self.config = config or {}
        
        # Performance optimizations
        self.model_cache = ModelCache(
            max_size=self.config.get('cache_size', 1000),
            ttl_seconds=self.config.get('cache_ttl', 300)
        )
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.enable_ensemble = self.config.get('enable_ensemble', True)
        self.monte_carlo_samples = self.config.get('monte_carlo_samples', 10)
        
        # Model warming
        self._warm_models()
        
        logger.info("Enhanced Risk Predictor initialized with caching and optimization")
    
    def _warm_models(self):
        """Warm up models with dummy data to improve first prediction performance"""
        logger.info("Warming up models...")
        
        try:
            for model_name, model in self.model_manager.models.items():
                model.eval()
                with torch.no_grad():
                    if model_name == 'satellite':
                        dummy_input = torch.randn(1, 3, 224, 224)
                    elif model_name in ['sentiment', 'news']:
                        dummy_input = torch.randint(0, 1000, (1, 512))
                    elif model_name == 'weather':
                        dummy_input = torch.randn(1, 30, 10)
                    elif model_name == 'economic':
                        dummy_input = torch.randn(1, 60, 15)
                    
                    _ = model(dummy_input)
            
            logger.info("Model warming completed")
        except Exception as e:
            logger.warning(f"Model warming failed: {e}")
    
    def predict_risk(self, timeframe_days: int = 30, include_uncertainty: bool = True) -> Dict[str, Any]:
        """Enhanced risk prediction with caching and uncertainty quantification"""
        start_time = time.time()
        
        # Check cache first
        cache_key = {'timeframe_days': timeframe_days, 'include_uncertainty': include_uncertainty}
        cached_result = self.model_cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Serving cached prediction for timeframe {timeframe_days} days")
            cached_result['cached'] = True
            cached_result['processing_time_ms'] = (time.time() - start_time) * 1000
            return cached_result
        
        logger.info(f"Generating new risk prediction for {timeframe_days} days ahead")
        
        # Generate predictions
        prediction_result = self._generate_enhanced_prediction(timeframe_days, include_uncertainty)
        
        # Cache the result
        result_dict = self._convert_to_dict(prediction_result)
        self.model_cache.put(cache_key, result_dict)
        
        total_time = (time.time() - start_time) * 1000
        result_dict['processing_time_ms'] = total_time
        result_dict['cached'] = False
        
        logger.info(f"Prediction completed in {total_time:.2f}ms")
        
        return result_dict
    
    def _generate_enhanced_prediction(self, timeframe_days: int, include_uncertainty: bool) -> RiskPredictionResult:
        """Generate enhanced prediction with uncertainty quantification"""
        inference_start = time.time()
        
        # Generate individual model predictions
        predictions = {}
        model_times = {}
        
        for model_name, model in self.model_manager.models.items():
            model_start = time.time()
            
            model.eval()
            with torch.no_grad():
                if model_name == 'satellite':
                    sample_input = torch.randn(1, 3, 224, 224)
                elif model_name in ['sentiment', 'news']:
                    sample_input = torch.randint(0, 1000, (1, 512))
                elif model_name == 'weather':
                    sample_input = torch.randn(1, 30, 10)
                elif model_name == 'economic':
                    sample_input = torch.randn(1, 60, 15)
                
                pred = model(sample_input)
                predictions[model_name] = pred
            
            model_times[model_name] = (time.time() - model_start) * 1000
        
        inference_time = (time.time() - inference_start) * 1000
        
        # Fusion with uncertainty quantification
        fusion_start = time.time()
        
        if include_uncertainty and self.enable_ensemble:
            # Monte Carlo ensemble for uncertainty estimation
            ensemble_predictions = []
            
            for _ in range(self.monte_carlo_samples):
                # Add small noise to inputs for uncertainty estimation
                noisy_predictions = {}
                for name, pred in predictions.items():
                    noise = torch.randn_like(pred) * 0.01
                    noisy_predictions[name] = pred + noise
                
                ensemble_risk = self.fusion_engine.fuse_predictions(noisy_predictions)
                ensemble_predictions.append(ensemble_risk)
            
            # Calculate uncertainty metrics
            risk_scores, confidence_intervals, uncertainty_metrics = self._calculate_uncertainty_metrics(ensemble_predictions)
        else:
            # Single prediction without uncertainty
            risk_scores = self.fusion_engine.fuse_predictions(predictions)
            confidence_intervals = {}
            uncertainty_metrics = {}
        
        fusion_time = (time.time() - fusion_start) * 1000
        
        # Calculate model agreement and contributions
        model_agreement = self._calculate_model_agreement(predictions)
        model_contributions = self._calculate_model_contributions(predictions, risk_scores)
        
        # Generate enhanced recommendations
        recommendations = self._generate_enhanced_recommendations(
            risk_scores, uncertainty_metrics, model_agreement, timeframe_days
        )
        
        # Create prediction metrics
        prediction_metrics = PredictionMetrics(
            processing_time_ms=inference_time + fusion_time,
            model_inference_time_ms=inference_time,
            fusion_time_ms=fusion_time,
            cache_hit=False,
            confidence_score=risk_scores.get('confidence', 0.8),
            model_agreement=model_agreement,
            prediction_uncertainty=uncertainty_metrics.get('overall_uncertainty', 0.1)
        )
        
        # Get model versions (simplified)
        model_versions = {name: "v1.0" for name in self.model_manager.models.keys()}
        
        return RiskPredictionResult(
            risk_scores=risk_scores,
            individual_predictions={k: v.mean().item() for k, v in predictions.items()},
            confidence_intervals=confidence_intervals,
            model_contributions=model_contributions,
            uncertainty_metrics=uncertainty_metrics,
            recommendations=recommendations,
            prediction_metrics=prediction_metrics,
            timestamp=datetime.now(),
            model_versions=model_versions
        )
    
    def _calculate_uncertainty_metrics(self, ensemble_predictions: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], Dict[str, float]]:
        """Calculate uncertainty metrics from ensemble predictions"""
        # Aggregate predictions
        aggregated = {}
        for key in ensemble_predictions[0].keys():
            values = [pred[key] for pred in ensemble_predictions]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            aggregated[key] = mean_val
            
        # Calculate confidence intervals (95%)
        confidence_intervals = {}
        uncertainty_metrics = {}
        
        for key in ensemble_predictions[0].keys():
            values = np.array([pred[key] for pred in ensemble_predictions])
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # 95% confidence interval
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            confidence_intervals[key] = (ci_lower, ci_upper)
            
            # Uncertainty metric (coefficient of variation)
            uncertainty_metrics[f"{key}_uncertainty"] = std_val / mean_val if mean_val != 0 else 0
        
        # Overall uncertainty
        uncertainty_metrics['overall_uncertainty'] = np.mean(list(uncertainty_metrics.values()))
        
        return aggregated, confidence_intervals, uncertainty_metrics
    
    def _calculate_model_agreement(self, predictions: Dict[str, torch.Tensor]) -> float:
        """Calculate agreement between model predictions"""
        if len(predictions) < 2:
            return 1.0
        
        # Convert predictions to normalized scores
        normalized_preds = []
        for pred in predictions.values():
            norm_pred = torch.softmax(pred.flatten(), dim=0)
            normalized_preds.append(norm_pred.mean().item())
        
        # Calculate coefficient of variation as agreement measure
        pred_array = np.array(normalized_preds)
        agreement = 1.0 - (np.std(pred_array) / np.mean(pred_array)) if np.mean(pred_array) != 0 else 0.0
        
        return max(0.0, min(1.0, agreement))
    
    def _calculate_model_contributions(self, predictions: Dict[str, torch.Tensor], risk_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate each model's contribution to final prediction"""
        contributions = {}
        total_weight = sum(pred.mean().item() for pred in predictions.values())
        
        if total_weight == 0:
            # Equal contributions if all predictions are zero
            equal_contrib = 1.0 / len(predictions)
            return {name: equal_contrib for name in predictions.keys()}
        
        for name, pred in predictions.items():
            weight = pred.mean().item() / total_weight
            contributions[name] = weight
        
        return contributions
    
    def _generate_enhanced_recommendations(self, risk_scores: Dict[str, float], 
                                         uncertainty_metrics: Dict[str, float], 
                                         model_agreement: float, 
                                         timeframe_days: int) -> List[str]:
        """Generate enhanced recommendations based on risk, uncertainty, and agreement"""
        recommendations = []
        overall_risk = risk_scores.get('overall_risk', 0.5)
        overall_uncertainty = uncertainty_metrics.get('overall_uncertainty', 0.1)
        
        # Risk-based recommendations
        if overall_risk > 0.8:
            recommendations.extend([
                "🚨 CRITICAL: Immediate supply chain diversification required",
                "⚡ Activate emergency supplier protocols within 24 hours",
                "📦 Increase inventory buffers for critical components by 50%"
            ])
        elif overall_risk > 0.6:
            recommendations.extend([
                "⚠️ HIGH: Review supplier contracts and contingency plans",
                "🔍 Consider alternative sourcing options",
                "📊 Monitor key suppliers with daily check-ins"
            ])
        elif overall_risk > 0.4:
            recommendations.extend([
                "📋 MEDIUM: Maintain current monitoring protocols",
                "📞 Prepare contingency communications",
                "🗺️ Review logistics routing options"
            ])
        else:
            recommendations.append("✅ LOW: Continue standard monitoring procedures")
        
        # Uncertainty-based recommendations
        if overall_uncertainty > 0.3:
            recommendations.append("🎯 High prediction uncertainty - collect additional data sources")
        
        # Model agreement recommendations
        if model_agreement < 0.6:
            recommendations.append("⚖️ Low model agreement - investigate conflicting signals")
        
        # Timeframe-specific recommendations
        if timeframe_days <= 7:
            recommendations.append("⏰ Short-term focus: Monitor daily operational metrics")
        elif timeframe_days <= 30:
            recommendations.append("📅 Medium-term focus: Review weekly supplier performance")
        else:
            recommendations.append("📈 Long-term focus: Strategic supplier relationship planning")
        
        return recommendations
    
    def _convert_to_dict(self, result: RiskPredictionResult) -> Dict[str, Any]:
        """Convert prediction result to dictionary format"""
        return {
            'timestamp': result.timestamp.isoformat(),
            'risk_scores': result.risk_scores,
            'individual_predictions': result.individual_predictions,
            'confidence_intervals': {k: list(v) for k, v in result.confidence_intervals.items()},
            'model_contributions': result.model_contributions,
            'uncertainty_metrics': result.uncertainty_metrics,
            'recommendations': result.recommendations,
            'prediction_metrics': {
                'processing_time_ms': result.prediction_metrics.processing_time_ms,
                'model_inference_time_ms': result.prediction_metrics.model_inference_time_ms,
                'fusion_time_ms': result.prediction_metrics.fusion_time_ms,
                'cache_hit': result.prediction_metrics.cache_hit,
                'confidence_score': result.prediction_metrics.confidence_score,
                'model_agreement': result.prediction_metrics.model_agreement,
                'prediction_uncertainty': result.prediction_metrics.prediction_uncertainty
            },
            'model_versions': result.model_versions
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.model_cache.get_stats()
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.model_cache.clear()
        logger.info("Prediction cache cleared")
    
    def batch_predict(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch prediction processing for improved efficiency"""
        logger.info(f"Processing batch of {len(requests)} predictions")
        
        results = []
        for request in requests:
            timeframe = request.get('timeframe_days', 30)
            include_uncertainty = request.get('include_uncertainty', True)
            result = self.predict_risk(timeframe, include_uncertainty)
            results.append(result)
        
        return results
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return {
            "cache_stats": self.get_cache_stats(),
            "model_count": len(self.model_manager.models),
            "fusion_engine_ready": hasattr(self, 'fusion_engine'),
            "ensemble_enabled": self.enable_ensemble,
            "monte_carlo_samples": self.monte_carlo_samples
        }
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
