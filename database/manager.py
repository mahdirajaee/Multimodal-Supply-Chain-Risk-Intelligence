"""
Database manager for Supply Chain Risk Intelligence System
Handles connections, migrations, and CRUD operations
"""

import os
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

from .models import Base, RiskPrediction, DataQuality, SystemMetrics, Alert, ModelVersion, APIUsage

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Comprehensive database manager with sync and async support"""
    
    def __init__(self, database_url: Optional[str] = None, async_database_url: Optional[str] = None):
        # Default to SQLite if no URL provided
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///supply_chain_risk.db')
        self.async_database_url = async_database_url or os.getenv('ASYNC_DATABASE_URL', 'sqlite+aiosqlite:///supply_chain_risk.db')
        
        # Create engines
        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        self.async_engine = create_async_engine(
            self.async_database_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Create session factories
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.AsyncSessionLocal = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self):
        """Async context manager for database sessions"""
        session = self.AsyncSessionLocal()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise
        finally:
            await session.close()
    
    # Risk Prediction Operations
    def save_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """Save a risk prediction to database"""
        with self.get_session() as session:
            prediction = RiskPrediction(
                location=prediction_data['location'],
                overall_risk=prediction_data['overall_risk'],
                satellite_risk=prediction_data.get('satellite_risk', 0.0),
                weather_risk=prediction_data.get('weather_risk', 0.0),
                economic_risk=prediction_data.get('economic_risk', 0.0),
                news_risk=prediction_data.get('news_risk', 0.0),
                social_risk=prediction_data.get('social_risk', 0.0),
                confidence=prediction_data.get('confidence', 0.0),
                uncertainty=prediction_data.get('uncertainty', 0.0),
                model_agreement=prediction_data.get('model_agreement', 0.0),
                satellite_data=prediction_data.get('satellite_data'),
                weather_data=prediction_data.get('weather_data'),
                economic_data=prediction_data.get('economic_data'),
                news_data=prediction_data.get('news_data'),
                social_data=prediction_data.get('social_data'),
                model_version=prediction_data.get('model_version', '1.0.0'),
                processing_time_ms=prediction_data.get('processing_time_ms'),
                data_quality_score=prediction_data.get('data_quality_score')
            )
            session.add(prediction)
            session.flush()
            return prediction.id
    
    def get_predictions_by_location(self, location: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent predictions for a location"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.get_session() as session:
            predictions = session.query(RiskPrediction).filter(
                RiskPrediction.location == location,
                RiskPrediction.timestamp >= cutoff_time
            ).order_by(RiskPrediction.timestamp.desc()).all()
            
            return [self._prediction_to_dict(p) for p in predictions]
    
    def get_risk_trends(self, location: str, days: int = 7) -> pd.DataFrame:
        """Get risk trends for analysis"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            predictions = session.query(RiskPrediction).filter(
                RiskPrediction.location == location,
                RiskPrediction.timestamp >= cutoff_time
            ).order_by(RiskPrediction.timestamp.asc()).all()
            
            data = []
            for p in predictions:
                data.append({
                    'timestamp': p.timestamp,
                    'overall_risk': p.overall_risk,
                    'satellite_risk': p.satellite_risk,
                    'weather_risk': p.weather_risk,
                    'economic_risk': p.economic_risk,
                    'news_risk': p.news_risk,
                    'social_risk': p.social_risk,
                    'confidence': p.confidence
                })
            
            return pd.DataFrame(data)
    
    # Data Quality Operations
    def save_data_quality(self, quality_data: Dict[str, Any]) -> int:
        """Save data quality metrics"""
        with self.get_session() as session:
            quality = DataQuality(
                data_source=quality_data['data_source'],
                completeness_score=quality_data['completeness_score'],
                accuracy_score=quality_data['accuracy_score'],
                consistency_score=quality_data['consistency_score'],
                timeliness_score=quality_data['timeliness_score'],
                validity_score=quality_data['validity_score'],
                overall_score=quality_data['overall_score'],
                issues_detected=quality_data.get('issues_detected', []),
                recommendations=quality_data.get('recommendations', []),
                records_processed=quality_data.get('records_processed'),
                processing_duration_ms=quality_data.get('processing_duration_ms')
            )
            session.add(quality)
            session.flush()
            return quality.id
    
    def get_data_quality_trends(self, data_source: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get data quality trends"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.get_session() as session:
            qualities = session.query(DataQuality).filter(
                DataQuality.data_source == data_source,
                DataQuality.timestamp >= cutoff_time
            ).order_by(DataQuality.timestamp.desc()).all()
            
            return [self._quality_to_dict(q) for q in qualities]
    
    # System Metrics Operations
    def save_system_metrics(self, metrics_data: Dict[str, Any]) -> int:
        """Save system performance metrics"""
        with self.get_session() as session:
            metrics = SystemMetrics(
                cpu_usage=metrics_data.get('cpu_usage'),
                memory_usage=metrics_data.get('memory_usage'),
                disk_usage=metrics_data.get('disk_usage'),
                network_io=metrics_data.get('network_io'),
                requests_per_minute=metrics_data.get('requests_per_minute'),
                avg_response_time_ms=metrics_data.get('avg_response_time_ms'),
                error_rate=metrics_data.get('error_rate'),
                cache_hit_rate=metrics_data.get('cache_hit_rate'),
                cache_size_mb=metrics_data.get('cache_size_mb'),
                prediction_latency_ms=metrics_data.get('prediction_latency_ms'),
                model_accuracy=metrics_data.get('model_accuracy'),
                model_drift_score=metrics_data.get('model_drift_score'),
                uptime_seconds=metrics_data.get('uptime_seconds'),
                health_status=metrics_data.get('health_status', 'healthy')
            )
            session.add(metrics)
            session.flush()
            return metrics.id
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the most recent system metrics"""
        with self.get_session() as session:
            metrics = session.query(SystemMetrics).order_by(
                SystemMetrics.timestamp.desc()
            ).first()
            
            return self._metrics_to_dict(metrics) if metrics else None
    
    # Alert Operations
    def create_alert(self, alert_data: Dict[str, Any]) -> int:
        """Create a new alert"""
        with self.get_session() as session:
            alert = Alert(
                alert_type=alert_data['alert_type'],
                severity=alert_data['severity'],
                title=alert_data['title'],
                message=alert_data['message'],
                location=alert_data.get('location'),
                data_source=alert_data.get('data_source'),
                metric_value=alert_data.get('metric_value'),
                threshold_value=alert_data.get('threshold_value'),
                details=alert_data.get('details')
            )
            session.add(alert)
            session.flush()
            return alert.id
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by severity"""
        with self.get_session() as session:
            query = session.query(Alert).filter(Alert.status == 'active')
            
            if severity:
                query = query.filter(Alert.severity == severity)
            
            alerts = query.order_by(Alert.timestamp.desc()).all()
            return [self._alert_to_dict(a) for a in alerts]
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert"""
        with self.get_session() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.status = 'acknowledged'
                alert.acknowledged_at = datetime.utcnow()
                return True
            return False
    
    def resolve_alert(self, alert_id: int) -> bool:
        """Resolve an alert"""
        with self.get_session() as session:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.status = 'resolved'
                alert.resolved_at = datetime.utcnow()
                return True
            return False
    
    # Analytics and Reporting
    def get_risk_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get risk summary statistics"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            # Risk distribution
            risk_stats = session.query(
                func.avg(RiskPrediction.overall_risk).label('avg_risk'),
                func.max(RiskPrediction.overall_risk).label('max_risk'),
                func.min(RiskPrediction.overall_risk).label('min_risk'),
                func.count(RiskPrediction.id).label('total_predictions')
            ).filter(RiskPrediction.timestamp >= cutoff_time).first()
            
            # Top risk locations
            top_risks = session.query(
                RiskPrediction.location,
                func.avg(RiskPrediction.overall_risk).label('avg_risk')
            ).filter(
                RiskPrediction.timestamp >= cutoff_time
            ).group_by(RiskPrediction.location).order_by(
                func.avg(RiskPrediction.overall_risk).desc()
            ).limit(10).all()
            
            return {
                'avg_risk': float(risk_stats.avg_risk or 0),
                'max_risk': float(risk_stats.max_risk or 0),
                'min_risk': float(risk_stats.min_risk or 0),
                'total_predictions': risk_stats.total_predictions or 0,
                'top_risk_locations': [
                    {'location': loc, 'avg_risk': float(risk)}
                    for loc, risk in top_risks
                ]
            }
    
    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Clean up old data to manage database size"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            # Count records to be deleted
            predictions_count = session.query(RiskPrediction).filter(
                RiskPrediction.timestamp < cutoff_time
            ).count()
            
            metrics_count = session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_time
            ).count()
            
            api_usage_count = session.query(APIUsage).filter(
                APIUsage.timestamp < cutoff_time
            ).count()
            
            # Delete old records
            session.query(RiskPrediction).filter(
                RiskPrediction.timestamp < cutoff_time
            ).delete()
            
            session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_time
            ).delete()
            
            session.query(APIUsage).filter(
                APIUsage.timestamp < cutoff_time
            ).delete()
            
            return {
                'predictions_deleted': predictions_count,
                'metrics_deleted': metrics_count,
                'api_usage_deleted': api_usage_count
            }
    
    # Helper methods
    def _prediction_to_dict(self, prediction: RiskPrediction) -> Dict[str, Any]:
        """Convert prediction model to dictionary"""
        return {
            'id': prediction.id,
            'timestamp': prediction.timestamp.isoformat(),
            'location': prediction.location,
            'overall_risk': prediction.overall_risk,
            'satellite_risk': prediction.satellite_risk,
            'weather_risk': prediction.weather_risk,
            'economic_risk': prediction.economic_risk,
            'news_risk': prediction.news_risk,
            'social_risk': prediction.social_risk,
            'confidence': prediction.confidence,
            'uncertainty': prediction.uncertainty,
            'model_agreement': prediction.model_agreement,
            'model_version': prediction.model_version,
            'processing_time_ms': prediction.processing_time_ms,
            'data_quality_score': prediction.data_quality_score
        }
    
    def _quality_to_dict(self, quality: DataQuality) -> Dict[str, Any]:
        """Convert quality model to dictionary"""
        return {
            'id': quality.id,
            'timestamp': quality.timestamp.isoformat(),
            'data_source': quality.data_source,
            'completeness_score': quality.completeness_score,
            'accuracy_score': quality.accuracy_score,
            'consistency_score': quality.consistency_score,
            'timeliness_score': quality.timeliness_score,
            'validity_score': quality.validity_score,
            'overall_score': quality.overall_score,
            'issues_detected': quality.issues_detected,
            'recommendations': quality.recommendations,
            'records_processed': quality.records_processed,
            'processing_duration_ms': quality.processing_duration_ms
        }
    
    def _metrics_to_dict(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Convert metrics model to dictionary"""
        return {
            'id': metrics.id,
            'timestamp': metrics.timestamp.isoformat(),
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage,
            'disk_usage': metrics.disk_usage,
            'network_io': metrics.network_io,
            'requests_per_minute': metrics.requests_per_minute,
            'avg_response_time_ms': metrics.avg_response_time_ms,
            'error_rate': metrics.error_rate,
            'cache_hit_rate': metrics.cache_hit_rate,
            'cache_size_mb': metrics.cache_size_mb,
            'prediction_latency_ms': metrics.prediction_latency_ms,
            'model_accuracy': metrics.model_accuracy,
            'model_drift_score': metrics.model_drift_score,
            'uptime_seconds': metrics.uptime_seconds,
            'health_status': metrics.health_status
        }
    
    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert alert model to dictionary"""
        return {
            'id': alert.id,
            'timestamp': alert.timestamp.isoformat(),
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'title': alert.title,
            'message': alert.message,
            'location': alert.location,
            'data_source': alert.data_source,
            'metric_value': alert.metric_value,
            'threshold_value': alert.threshold_value,
            'status': alert.status,
            'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
            'details': alert.details
        }

# Global database manager instance
db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager
