#!/usr/bin/env python3
"""
Enhanced FastAPI Server with Performance Optimizations
Includes caching, rate limiting, authentication, monitoring, and async processing
"""

import asyncio
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import redis
import aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
from cachetools import TTLCache

from .predictor import RiskPredictor
from .middleware import RateLimitMiddleware, LoggingMiddleware, MetricsMiddleware

logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
CACHE_HITS = Counter('cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Cache misses')

# In-memory cache for small datasets
memory_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes TTL

class EnhancedRiskPredictionRequest(BaseModel):
    timeframe_days: int = Field(default=30, ge=1, le=365, description="Prediction timeframe in days")
    include_recommendations: bool = Field(default=True, description="Include recommendations in response")
    confidence_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID for tracking")
    
class EnhancedRiskPredictionResponse(BaseModel):
    timestamp: str
    timeframe_days: int
    risk_scores: Dict[str, float]
    individual_predictions: Dict[str, float]
    recommendations: List[str]
    confidence_score: float
    prediction_id: str
    correlation_id: Optional[str] = None
    processing_time_ms: float
    cached: bool = False
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    memory_usage_mb: float
    active_connections: int
    cache_stats: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str
    correlation_id: Optional[str] = None

# Enhanced FastAPI app
app = FastAPI(
    title="Enhanced Supply Chain Risk Intelligence API",
    description="Production-ready multimodal supply chain risk prediction system with caching, rate limiting, and monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8502", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "yourdomain.com"])
app.add_middleware(RateLimitMiddleware, calls=100, period=60)  # 100 requests per minute
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Global state
predictor_instance = None
redis_client = None
start_time = time.time()

class SecurityManager:
    """Enhanced security for API authentication and authorization"""
    
    def __init__(self):
        self.valid_tokens = {
            "demo_token_123": {"user": "demo", "permissions": ["read", "predict"]},
            "admin_token_456": {"user": "admin", "permissions": ["read", "predict", "retrain", "admin"]}
        }
    
    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Security(security)):
        token = credentials.credentials
        if token not in self.valid_tokens:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return self.valid_tokens[token]

security_manager = SecurityManager()

class CacheManager:
    """Enhanced caching with Redis fallback and memory cache"""
    
    def __init__(self):
        self.memory_cache = memory_cache
        
    async def get_redis_client(self):
        global redis_client
        if redis_client is None:
            try:
                redis_client = await aioredis.from_url("redis://localhost:6379")
            except Exception as e:
                logger.warning(f"Redis not available, using memory cache only: {e}")
                redis_client = None
        return redis_client
    
    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a consistent cache key from request data"""
        cache_str = f"{request_data['timeframe_days']}_{request_data['confidence_threshold']}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    async def get_cached_prediction(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available"""
        cache_key = self._generate_cache_key(request_data)
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            CACHE_HITS.inc()
            cached_result = self.memory_cache[cache_key]
            cached_result['cached'] = True
            return cached_result
        
        # Try Redis cache
        redis_client = await self.get_redis_client()
        if redis_client:
            try:
                cached_data = await redis_client.get(f"prediction:{cache_key}")
                if cached_data:
                    CACHE_HITS.inc()
                    import json
                    result = json.loads(cached_data)
                    result['cached'] = True
                    # Also store in memory cache for faster access
                    self.memory_cache[cache_key] = result
                    return result
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        CACHE_MISSES.inc()
        return None
    
    async def cache_prediction(self, request_data: Dict[str, Any], prediction: Dict[str, Any]):
        """Cache prediction result"""
        cache_key = self._generate_cache_key(request_data)
        
        # Store in memory cache
        self.memory_cache[cache_key] = prediction.copy()
        
        # Store in Redis cache
        redis_client = await self.get_redis_client()
        if redis_client:
            try:
                import json
                await redis_client.setex(
                    f"prediction:{cache_key}", 
                    300,  # 5 minutes TTL
                    json.dumps(prediction, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis cache store error: {e}")

cache_manager = CacheManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Enhanced Supply Chain Risk Intelligence API")
    await cache_manager.get_redis_client()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global redis_client
    if redis_client:
        await redis_client.close()
    logger.info("API shutdown complete")

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Enhanced Supply Chain Risk Intelligence API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Multimodal AI predictions",
            "Real-time caching",
            "Rate limiting",
            "Authentication",
            "Metrics monitoring",
            "Async processing"
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def enhanced_health_check():
    """Enhanced health check with system metrics"""
    import psutil
    
    uptime = time.time() - start_time
    memory_info = psutil.virtual_memory()
    
    cache_stats = {
        "memory_cache_size": len(memory_cache),
        "memory_cache_maxsize": memory_cache.maxsize,
        "redis_available": redis_client is not None
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        uptime_seconds=uptime,
        memory_usage_mb=memory_info.used / 1024 / 1024,
        active_connections=int(ACTIVE_CONNECTIONS._value.get()),
        cache_stats=cache_stats
    )

@app.post("/predict", response_model=EnhancedRiskPredictionResponse, tags=["Prediction"])
async def enhanced_predict_risk(
    request: EnhancedRiskPredictionRequest,
    user: Dict = Depends(security_manager.verify_token)
):
    """Enhanced risk prediction with caching and async processing"""
    start_time = time.time()
    
    # Generate correlation ID if not provided
    correlation_id = request.correlation_id or str(uuid.uuid4())
    
    logger.info(f"Risk prediction request [{correlation_id}] from user {user['user']}")
    
    try:
        if predictor_instance is None:
            raise HTTPException(status_code=503, detail="Predictor not initialized")
        
        # Check permissions
        if "predict" not in user["permissions"]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Try to get cached result
        request_data = request.dict()
        cached_result = await cache_manager.get_cached_prediction(request_data)
        
        if cached_result:
            logger.info(f"Serving cached prediction [{correlation_id}]")
            cached_result['correlation_id'] = correlation_id
            cached_result['processing_time_ms'] = (time.time() - start_time) * 1000
            return EnhancedRiskPredictionResponse(**cached_result)
        
        # Generate new prediction
        prediction = await asyncio.get_event_loop().run_in_executor(
            None, 
            predictor_instance.predict_risk, 
            request.timeframe_days
        )
        
        # Add enhancement fields
        prediction['prediction_id'] = str(uuid.uuid4())
        prediction['correlation_id'] = correlation_id
        prediction['confidence_score'] = prediction['risk_scores']['confidence']
        prediction['processing_time_ms'] = (time.time() - start_time) * 1000
        prediction['cached'] = False
        
        # Filter recommendations if needed
        if not request.include_recommendations:
            prediction['recommendations'] = []
        
        # Cache the result
        await cache_manager.cache_prediction(request_data, prediction)
        
        logger.info(f"Prediction completed [{correlation_id}] in {prediction['processing_time_ms']:.2f}ms")
        
        return EnhancedRiskPredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Prediction error [{correlation_id}]: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=ErrorResponse(
                error="prediction_failed",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                correlation_id=correlation_id
            ).dict()
        )

@app.get("/models/status", tags=["Models"])
async def enhanced_model_status(user: Dict = Depends(security_manager.verify_token)):
    """Enhanced model status with detailed information"""
    if predictor_instance is None:
        return {"status": "not_initialized"}
    
    models = predictor_instance.model_manager.models
    model_info = {}
    
    for name, model in models.items():
        # Get model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info[name] = {
            "loaded": True,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": next(model.parameters()).device.type if total_params > 0 else "cpu"
        }
    
    return {
        "status": "ready",
        "models": model_info,
        "fusion_engine_loaded": hasattr(predictor_instance, 'fusion_engine'),
        "last_updated": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/models/retrain", tags=["Models"])
async def enhanced_retrain_models(
    background_tasks: BackgroundTasks,
    user: Dict = Depends(security_manager.verify_token)
):
    """Enhanced model retraining with permission check"""
    if "retrain" not in user["permissions"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions for model retraining")
    
    correlation_id = str(uuid.uuid4())
    background_tasks.add_task(_retrain_models_background, correlation_id, user['user'])
    
    return {
        "message": "Model retraining initiated",
        "correlation_id": correlation_id,
        "status": "processing",
        "initiated_by": user['user']
    }

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/cache/stats", tags=["Monitoring"])
async def get_cache_stats(user: Dict = Depends(security_manager.verify_token)):
    """Get cache statistics"""
    if "admin" not in user["permissions"]:
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    stats = {
        "memory_cache": {
            "size": len(memory_cache),
            "maxsize": memory_cache.maxsize,
            "ttl": memory_cache.ttl
        },
        "cache_hits": CACHE_HITS._value.get(),
        "cache_misses": CACHE_MISSES._value.get(),
        "hit_ratio": CACHE_HITS._value.get() / (CACHE_HITS._value.get() + CACHE_MISSES._value.get()) if (CACHE_HITS._value.get() + CACHE_MISSES._value.get()) > 0 else 0
    }
    
    if redis_client:
        try:
            redis_info = await redis_client.info()
            stats["redis"] = {
                "connected": True,
                "memory_usage": redis_info.get("used_memory_human"),
                "connected_clients": redis_info.get("connected_clients")
            }
        except Exception as e:
            stats["redis"] = {"connected": False, "error": str(e)}
    else:
        stats["redis"] = {"connected": False}
    
    return stats

@app.post("/cache/clear", tags=["Monitoring"])
async def clear_cache(user: Dict = Depends(security_manager.verify_token)):
    """Clear all caches"""
    if "admin" not in user["permissions"]:
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    # Clear memory cache
    memory_cache.clear()
    
    # Clear Redis cache
    if redis_client:
        try:
            await redis_client.flushdb()
        except Exception as e:
            logger.warning(f"Failed to clear Redis cache: {e}")
    
    return {"message": "Cache cleared successfully"}

async def _retrain_models_background(correlation_id: str, user: str):
    """Background task for model retraining"""
    logger.info(f"Background model retraining started [{correlation_id}] by {user}")
    
    try:
        # Simulate retraining process
        await asyncio.sleep(10)  # Placeholder for actual retraining
        logger.info(f"Model retraining completed [{correlation_id}]")
    except Exception as e:
        logger.error(f"Model retraining failed [{correlation_id}]: {e}")

def start_enhanced_server(predictor: RiskPredictor, config: Dict[str, Any]):
    """Start the enhanced server with all optimizations"""
    global predictor_instance
    predictor_instance = predictor
    
    uvicorn.run(
        app,
        host=config.get('host', '0.0.0.0'),
        port=config.get('port', 8000),
        workers=1,  # Use 1 worker for shared state
        log_level="info",
        access_log=True
    )
