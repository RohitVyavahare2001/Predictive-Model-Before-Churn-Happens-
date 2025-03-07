from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import joblib
import os
from datetime import datetime
import logging
from churn_prediction_pipeline import ChurnPredictionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Churn Prediction API", version="1.1.0")

# Load model on startup
model_path = os.getenv("MODEL_PATH", "churn_prediction_model.joblib")
model_pipeline = ChurnPredictionPipeline()
try:
    model_pipeline.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model_pipeline = None

# Define input schemas
class CustomerFeatures(BaseModel):
    customer_id: str
    tenure: float
    monthly_charges: float
    total_charges: float
    avg_session_length: float
    avg_session_count_monthly: float
    days_since_last_activity: float
    support_tickets_count: int
    items_purchased_6m: int
    avg_purchase_value: float
    contract_type: str
    payment_method: str
    subscription_tier: str
    has_premium_support: bool
    has_family_plan: bool
    has_multiple_devices: bool

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]
    risk_threshold: float = Field(0.5, ge=0, le=1)

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    is_high_risk: bool
    prediction_time: str

@app.post("/predict", response_model=List[PredictionResponse])
def predict_churn(request: BatchPredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = pd.DataFrame([customer.dict() for customer in request.customers])
    customer_ids = df.pop("customer_id")
    churn_probs = model_pipeline.predict(df)
    
    results = [
        PredictionResponse(
            customer_id=customer_ids.iloc[i],
            churn_probability=float(prob),
            is_high_risk=prob >= request.risk_threshold,
            prediction_time=datetime.now().isoformat()
        ) for i, prob in enumerate(churn_probs)
    ]
    return results

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}
