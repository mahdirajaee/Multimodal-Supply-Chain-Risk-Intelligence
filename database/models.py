"""
Database models for Supply Chain Risk Intelligence System
Stores historical data, predictions, and system metrics
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional

Base = declarative_base()

class RiskPrediction(Base):
    """Store historical risk predictions and their context"""
    __tablename__ = 'risk_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    location = Column(String(255), nullable=False)
    
    # Risk scores
    overall_risk = Column(Float, nullable=False)
    satellite_risk = Column(Float, nullable=False)
    weather_risk = Column(Float, nullable=False)
    economic_risk = Column(Float, nullable=False)
    news_risk = Column(Float, nullable=False)
    social_risk = Column(Float, nullable=False)
    
    # Confidence and uncertainty
    confidence = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    model_agreement = Column(Float, nullable=False)
    
    # Raw data snapshots (JSON)
    satellite_data = Column(JSON)
    weather_data = Column(JSON)
    economic_data = Column(JSON)
    news_data = Column(JSON)
    social_data = Column(JSON)
    
    # Metadata
    model_version = Column(String(50), nullable=False)
    processing_time_ms = Column(Float)
    data_quality_score = Column(Float)
    
    # Create indexes for common queries
    __table_args__ = (
        Index('idx_timestamp_location', 'timestamp', 'location'),
        Index('idx_overall_risk', 'overall_risk'),
        Index('idx_timestamp', 'timestamp'),
    )

class DataQuality(Base):
    """Track data quality metrics over time"""
    __tablename__ = 'data_quality'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    data_source = Column(String(100), nullable=False)  # satellite, weather, economic, news, social
    
    # Quality metrics (0-1 scale)
    completeness_score = Column(Float, nullable=False)
    accuracy_score = Column(Float, nullable=False)
    consistency_score = Column(Float, nullable=False)
    timeliness_score = Column(Float, nullable=False)
    validity_score = Column(Float, nullable=False)
    overall_score = Column(Float, nullable=False)
    
    # Issues and recommendations
    issues_detected = Column(JSON)  # List of issues
    recommendations = Column(JSON)  # List of recommendations
    
    # Metadata
    records_processed = Column(Integer)
    processing_duration_ms = Column(Float)
    
    __table_args__ = (
        Index('idx_data_source_timestamp', 'data_source', 'timestamp'),
        Index('idx_overall_score', 'overall_score'),
    )

class SystemMetrics(Base):
    """Store system performance and health metrics"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Performance metrics
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)
    network_io = Column(Float)
    
    # API metrics
    requests_per_minute = Column(Integer)
    avg_response_time_ms = Column(Float)
    error_rate = Column(Float)
    
    # Cache metrics
    cache_hit_rate = Column(Float)
    cache_size_mb = Column(Float)
    
    # ML model metrics
    prediction_latency_ms = Column(Float)
    model_accuracy = Column(Float)
    model_drift_score = Column(Float)
    
    # Uptime and health
    uptime_seconds = Column(Integer)
    health_status = Column(String(20))  # healthy, degraded, critical
    
    __table_args__ = (
        Index('idx_timestamp_health', 'timestamp', 'health_status'),
    )

class Alert(Base):
    """Store system alerts and notifications"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # risk, quality, system, security
    severity = Column(String(20), nullable=False)    # low, medium, high, critical
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # Context
    location = Column(String(255))
    data_source = Column(String(100))
    metric_value = Column(Float)
    threshold_value = Column(Float)
    
    # Status
    status = Column(String(20), default='active')  # active, acknowledged, resolved
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Metadata
    details = Column(JSON)  # Additional context
    
    __table_args__ = (
        Index('idx_severity_status', 'severity', 'status'),
        Index('idx_alert_type_timestamp', 'alert_type', 'timestamp'),
    )

class ModelVersion(Base):
    """Track ML model versions and performance"""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Model identification
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # satellite, weather, economic, news, social, fusion
    
    # Model artifacts
    model_path = Column(String(500), nullable=False)
    config_path = Column(String(500))
    checksum = Column(String(64))  # SHA256 hash
    
    # Performance metrics
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float)
    test_accuracy = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    
    # Training details
    training_data_size = Column(Integer)
    training_duration_minutes = Column(Float)
    hyperparameters = Column(JSON)
    
    # Status
    is_active = Column(Boolean, default=False)
    deployment_status = Column(String(20), default='created')  # created, testing, deployed, retired
    
    __table_args__ = (
        Index('idx_model_name_version', 'model_name', 'version'),
        Index('idx_is_active', 'is_active'),
    )

class APIUsage(Base):
    """Track API usage patterns and user behavior"""
    __tablename__ = 'api_usage'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Request details
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    user_id = Column(String(100))
    user_role = Column(String(50))
    
    # Request context
    location_requested = Column(String(255))
    request_size_bytes = Column(Integer)
    response_size_bytes = Column(Integer)
    
    # Performance
    response_time_ms = Column(Float, nullable=False)
    status_code = Column(Integer, nullable=False)
    
    # Client info
    user_agent = Column(String(500))
    ip_address = Column(String(45))  # IPv6 compatible
    
    # Caching
    cache_hit = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_endpoint_timestamp', 'endpoint', 'timestamp'),
        Index('idx_user_id_timestamp', 'user_id', 'timestamp'),
        Index('idx_status_code', 'status_code'),
    )
