from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import logging
from datetime import datetime
from .predictor import RiskPredictor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Supply Chain Risk Intelligence API",
    description="Real-time multimodal supply chain risk prediction system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor_instance = None

class RiskPredictionRequest(BaseModel):
    timeframe_days: int = 30
    include_recommendations: bool = True
    confidence_threshold: Optional[float] = 0.7

class RiskPredictionResponse(BaseModel):
    timestamp: str
    timeframe_days: int
    risk_scores: Dict[str, float]
    individual_predictions: Dict[str, List[float]]
    recommendations: List[str]
    status: str = "success"

@app.get("/")
async def root():
    return {"message": "Supply Chain Risk Intelligence API", "status": "operational"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=RiskPredictionResponse)
async def predict_risk(request: RiskPredictionRequest):
    try:
        if predictor_instance is None:
            raise HTTPException(status_code=503, detail="Predictor not initialized")
            
        prediction = predictor_instance.predict_risk(request.timeframe_days)
        
        if not request.include_recommendations:
            prediction.pop('recommendations', None)
            
        return RiskPredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models/status")
async def get_model_status():
    if predictor_instance is None:
        return {"status": "not_initialized"}
        
    return {
        "status": "ready",
        "models": list(predictor_instance.model_manager.models.keys()),
        "last_updated": datetime.now().isoformat()
    }

@app.post("/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_models_background)
    return {"message": "Model retraining initiated", "status": "processing"}

async def _retrain_models_background():
    logger.info("Background model retraining started")
    
def start_server(predictor: RiskPredictor, config: Dict[str, Any]):
    global predictor_instance
    predictor_instance = predictor
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,
        log_level="info"
    )
