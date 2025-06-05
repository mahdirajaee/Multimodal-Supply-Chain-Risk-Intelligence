#!/usr/bin/env python3
"""
Enhanced Data Manager with Quality Checks, Validation, and Performance Optimizations
Includes data validation, quality scoring, automated cleanup, and async processing
"""

import asyncio
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass

from .satellite_data import SatelliteDataLoader
from .social_data import SocialDataLoader
from .weather_data import WeatherDataLoader
from .economic_data import EconomicDataLoader
from .news_data import NewsDataLoader

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics for monitoring"""
    completeness: float  # Percentage of non-null values
    accuracy: float     # Data accuracy score
    consistency: float  # Data consistency score
    timeliness: float   # Data freshness score
    validity: float     # Data format validity score
    overall_score: float
    timestamp: datetime
    issues: List[str]

@dataclass
class DataValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_metrics: DataQualityMetrics
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]

class DataQualityValidator:
    """Advanced data quality validation and scoring"""
    
    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.95,
            'accuracy': 0.90,
            'consistency': 0.85,
            'timeliness': 0.80,
            'validity': 0.95
        }
    
    def validate_tensor_data(self, data: torch.Tensor, expected_shape: Optional[Tuple] = None) -> DataValidationResult:
        """Validate tensor data quality"""
        errors = []
        warnings = []
        recommendations = []
        issues = []
        
        # Check for NaN values
        nan_count = torch.isnan(data).sum().item()
        total_elements = data.numel()
        completeness = 1.0 - (nan_count / total_elements) if total_elements > 0 else 0.0
        
        if completeness < self.quality_thresholds['completeness']:
            errors.append(f"High NaN rate: {nan_count}/{total_elements} ({(1-completeness)*100:.2f}%)")
            issues.append("missing_values")
        
        # Check for infinite values
        inf_count = torch.isinf(data).sum().item()
        if inf_count > 0:
            errors.append(f"Found {inf_count} infinite values")
            issues.append("infinite_values")
        
        # Check data range and outliers
        if data.dtype in [torch.float32, torch.float64]:
            mean_val = data.mean().item()
            std_val = data.std().item()
            
            # Check for extreme outliers (beyond 4 standard deviations)
            outlier_mask = torch.abs(data - mean_val) > 4 * std_val
            outlier_count = outlier_mask.sum().item()
            
            if outlier_count > total_elements * 0.05:  # More than 5% outliers
                warnings.append(f"High outlier rate: {outlier_count} values beyond 4σ")
                issues.append("outliers")
        
        # Check expected shape
        validity = 1.0
        if expected_shape and data.shape != expected_shape:
            errors.append(f"Shape mismatch: expected {expected_shape}, got {data.shape}")
            validity = 0.0
            issues.append("shape_mismatch")
        
        # Calculate quality scores
        accuracy = 1.0 - (outlier_count / total_elements) if total_elements > 0 else 0.0
        consistency = 1.0 if std_val > 0 and not torch.isnan(torch.tensor(std_val)) else 0.0
        timeliness = 1.0  # Assume fresh data for now
        
        # Overall score (weighted average)
        weights = {'completeness': 0.3, 'accuracy': 0.25, 'consistency': 0.2, 'timeliness': 0.15, 'validity': 0.1}
        overall_score = (
            completeness * weights['completeness'] +
            accuracy * weights['accuracy'] +
            consistency * weights['consistency'] +
            timeliness * weights['timeliness'] +
            validity * weights['validity']
        )
        
        # Generate recommendations
        if completeness < 0.9:
            recommendations.append("Improve data collection to reduce missing values")
        if accuracy < 0.85:
            recommendations.append("Implement outlier detection and cleaning")
        if consistency < 0.8:
            recommendations.append("Review data normalization procedures")
        
        quality_metrics = DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            overall_score=overall_score,
            timestamp=datetime.now(),
            issues=issues
        )
        
        is_valid = overall_score >= 0.8 and len(errors) == 0
        
        return DataValidationResult(
            is_valid=is_valid,
            quality_metrics=quality_metrics,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )

class EnhancedDataManager:
    """Enhanced data manager with quality checks, caching, and async processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_validator = DataQualityValidator()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize data loaders
        self.loaders = {
            'satellite': SatelliteDataLoader(config['satellite']),
            'social': SocialDataLoader(config['social']),
            'weather': WeatherDataLoader(config['weather']),
            'economic': EconomicDataLoader(config['economic']),
            'news': NewsDataLoader(config['news'])
        }
        
        # Data quality tracking
        self.quality_history = {}
        self.last_validation = {}
        
        # Cache settings
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour default
        self.enable_quality_checks = config.get('enable_quality_checks', True)
        
        logger.info("Enhanced Data Manager initialized with quality validation")
    
    async def prepare_training_data_async(self) -> Dict[str, DataValidationResult]:
        """Asynchronously prepare training data with quality validation"""
        logger.info("Preparing training data with quality validation")
        
        tasks = {}
        for name, loader in self.loaders.items():
            tasks[name] = asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._load_and_validate_data,
                name,
                loader
            )
        
        results = {}
        for name, task in tasks.items():
            results[name] = await task
        
        return results
    
    def _load_and_validate_data(self, name: str, loader) -> DataValidationResult:
        """Load data and perform quality validation"""
        try:
            logger.info(f"Loading and validating {name} data")
            
            # Load historical data
            loader.load_historical_data()
            
            # Get processed data
            data = loader.get_processed_data()
            
            # Validate each tensor in the data
            validation_results = []
            
            for key, tensor in data.items():
                if isinstance(tensor, torch.Tensor):
                    result = self.quality_validator.validate_tensor_data(tensor)
                    validation_results.append(result)
                    
                    logger.info(f"{name}.{key} quality score: {result.quality_metrics.overall_score:.3f}")
                    
                    if not result.is_valid:
                        logger.warning(f"{name}.{key} validation failed: {result.errors}")
            
            # Aggregate results
            if validation_results:
                overall_score = np.mean([r.quality_metrics.overall_score for r in validation_results])
                all_errors = []
                all_warnings = []
                all_recommendations = []
                
                for result in validation_results:
                    all_errors.extend(result.errors)
                    all_warnings.extend(result.warnings)
                    all_recommendations.extend(result.recommendations)
                
                aggregate_quality = DataQualityMetrics(
                    completeness=np.mean([r.quality_metrics.completeness for r in validation_results]),
                    accuracy=np.mean([r.quality_metrics.accuracy for r in validation_results]),
                    consistency=np.mean([r.quality_metrics.consistency for r in validation_results]),
                    timeliness=np.mean([r.quality_metrics.timeliness for r in validation_results]),
                    validity=np.mean([r.quality_metrics.validity for r in validation_results]),
                    overall_score=overall_score,
                    timestamp=datetime.now(),
                    issues=list(set([issue for r in validation_results for issue in r.quality_metrics.issues]))
                )
                
                final_result = DataValidationResult(
                    is_valid=overall_score >= 0.8 and len(all_errors) == 0,
                    quality_metrics=aggregate_quality,
                    errors=list(set(all_errors)),
                    warnings=list(set(all_warnings)),
                    recommendations=list(set(all_recommendations))
                )
            else:
                # No tensor data found
                final_result = DataValidationResult(
                    is_valid=False,
                    quality_metrics=DataQualityMetrics(0, 0, 0, 0, 0, 0, datetime.now(), ["no_data"]),
                    errors=["No tensor data found"],
                    warnings=[],
                    recommendations=["Check data loading implementation"]
                )
            
            # Store validation results
            self.last_validation[name] = final_result
            if name not in self.quality_history:
                self.quality_history[name] = []
            self.quality_history[name].append(final_result.quality_metrics)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error loading {name} data: {e}")
            error_result = DataValidationResult(
                is_valid=False,
                quality_metrics=DataQualityMetrics(0, 0, 0, 0, 0, 0, datetime.now(), ["load_error"]),
                errors=[f"Data loading failed: {str(e)}"],
                warnings=[],
                recommendations=["Check data source configuration and connectivity"]
            )
            self.last_validation[name] = error_result
            return error_result
    
    async def get_real_time_data_async(self) -> Dict[str, Any]:
        """Asynchronously fetch real-time data with quality checks"""
        logger.info("Fetching real-time data with quality validation")
        
        tasks = {}
        for name, loader in self.loaders.items():
            tasks[name] = asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._fetch_current_data,
                name,
                loader
            )
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Failed to fetch {name} data: {e}")
                results[name] = None
        
        return results
    
    def _fetch_current_data(self, name: str, loader):
        """Fetch current data with validation"""
        try:
            data = loader.get_current_data()
            
            if self.enable_quality_checks and isinstance(data, torch.Tensor):
                validation = self.quality_validator.validate_tensor_data(data)
                if not validation.is_valid:
                    logger.warning(f"Real-time {name} data quality issues: {validation.errors}")
                    return None
            
            return data
        except Exception as e:
            logger.error(f"Error fetching current {name} data: {e}")
            return None
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "sources": {},
            "overall_health": "unknown"
        }
        
        if not self.last_validation:
            report["overall_health"] = "no_data"
            return report
        
        total_score = 0
        valid_sources = 0
        
        for name, validation in self.last_validation.items():
            source_report = {
                "is_valid": validation.is_valid,
                "quality_score": validation.quality_metrics.overall_score,
                "completeness": validation.quality_metrics.completeness,
                "accuracy": validation.quality_metrics.accuracy,
                "consistency": validation.quality_metrics.consistency,
                "timeliness": validation.quality_metrics.timeliness,
                "validity": validation.quality_metrics.validity,
                "issues": validation.quality_metrics.issues,
                "errors": validation.errors,
                "warnings": validation.warnings,
                "recommendations": validation.recommendations,
                "last_check": validation.quality_metrics.timestamp.isoformat()
            }
            
            report["sources"][name] = source_report
            
            if validation.is_valid:
                total_score += validation.quality_metrics.overall_score
                valid_sources += 1
        
        # Calculate overall health
        if valid_sources == 0:
            report["overall_health"] = "critical"
        else:
            avg_score = total_score / valid_sources
            if avg_score >= 0.9:
                report["overall_health"] = "excellent"
            elif avg_score >= 0.8:
                report["overall_health"] = "good"
            elif avg_score >= 0.7:
                report["overall_health"] = "fair"
            else:
                report["overall_health"] = "poor"
        
        report["average_quality_score"] = total_score / valid_sources if valid_sources > 0 else 0
        report["valid_sources"] = valid_sources
        report["total_sources"] = len(self.last_validation)
        
        return report
    
    def clean_old_cache(self, max_age_hours: int = 24):
        """Clean old cached data"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for name in self.quality_history:
            self.quality_history[name] = [
                metric for metric in self.quality_history[name]
                if metric.timestamp > cutoff_time
            ]
        
        logger.info(f"Cleaned cache data older than {max_age_hours} hours")
    
    async def sync_data_sources_async(self):
        """Asynchronously sync all data sources"""
        logger.info("Syncing data sources")
        
        tasks = []
        for name, loader in self.loaders.items():
            task = asyncio.get_event_loop().run_in_executor(
                self.executor,
                loader.sync_cache
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        logger.info("Data source sync completed")
    
    def get_data_freshness_status(self) -> Dict[str, Any]:
        """Get data freshness status for all sources"""
        status = {}
        now = datetime.now()
        
        for name, validation in self.last_validation.items():
            if validation:
                age_minutes = (now - validation.quality_metrics.timestamp).total_seconds() / 60
                
                if age_minutes <= 5:
                    freshness = "very_fresh"
                elif age_minutes <= 30:
                    freshness = "fresh"
                elif age_minutes <= 60:
                    freshness = "moderate"
                elif age_minutes <= 240:
                    freshness = "stale"
                else:
                    freshness = "very_stale"
                
                status[name] = {
                    "freshness": freshness,
                    "age_minutes": int(age_minutes),
                    "last_update": validation.quality_metrics.timestamp.isoformat()
                }
            else:
                status[name] = {
                    "freshness": "unknown",
                    "age_minutes": -1,
                    "last_update": None
                }
        
        return status
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
