from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for ML model
model = None
scaler = None
feature_columns = None
class_labels = None
model_metrics = None

# Data paths
DATA_PATH = ROOT_DIR / 'data' / 'train_dataset.csv'
MODEL_PATH = ROOT_DIR / 'models' / 'bean_classifier.joblib'
SCALER_PATH = ROOT_DIR / 'models' / 'scaler.joblib'

# Ensure directories exist
(ROOT_DIR / 'models').mkdir(exist_ok=True)
(ROOT_DIR / 'static').mkdir(exist_ok=True)

# Pydantic Models
class DatasetInfo(BaseModel):
    total_samples: int
    total_features: int
    class_distribution: Dict[str, int]
    feature_names: List[str]
    class_names: List[str]

class TrainResponse(BaseModel):
    success: bool
    accuracy: float
    classification_report: Dict[str, Any]
    message: str

class PredictionInput(BaseModel):
    Area: float
    Perimeter: float
    MajorAxisLength: float
    MinorAxisLength: float
    AspectRation: float
    Eccentricity: float
    ConvexArea: float
    EquivDiameter: float
    Extent: float
    Solidity: float
    roundness: float
    Compactness: float
    ShapeFactor1: float
    ShapeFactor2: float
    ShapeFactor3: float
    ShapeFactor4: float

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

class StatisticsResponse(BaseModel):
    feature_statistics: Dict[str, Dict[str, float]]
    class_distribution: Dict[str, int]
    total_samples: int

# Helper functions
def load_dataset():
    """Load the dry bean dataset"""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

def train_model_internal():
    """Train the ML model"""
    global model, scaler, feature_columns, class_labels, model_metrics
    
    df = load_dataset()
    if df is None:
        return None
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'Class']
    X = df[feature_columns]
    y = df['Class']
    class_labels = y.unique().tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save models
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    model_metrics = {
        'accuracy': accuracy,
        'report': report
    }
    
    return {
        'accuracy': accuracy,
        'report': report
    }

def load_trained_model():
    """Load pre-trained model if exists"""
    global model, scaler, feature_columns, class_labels
    
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        df = load_dataset()
        if df is not None:
            feature_columns = [col for col in df.columns if col != 'Class']
            class_labels = df['Class'].unique().tolist()
        return True
    return False

# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "Dry Bean Classification API"}

@api_router.get("/dataset-info", response_model=DatasetInfo)
async def get_dataset_info():
    """Get information about the dataset"""
    df = load_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    feature_cols = [col for col in df.columns if col != 'Class']
    class_dist = df['Class'].value_counts().to_dict()
    
    return DatasetInfo(
        total_samples=len(df),
        total_features=len(feature_cols),
        class_distribution=class_dist,
        feature_names=feature_cols,
        class_names=list(class_dist.keys())
    )

@api_router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """Get dataset statistics"""
    df = load_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    feature_cols = [col for col in df.columns if col != 'Class']
    stats = {}
    for col in feature_cols:
        stats[col] = {
            'mean': round(df[col].mean(), 4),
            'std': round(df[col].std(), 4),
            'min': round(df[col].min(), 4),
            'max': round(df[col].max(), 4)
        }
    
    return StatisticsResponse(
        feature_statistics=stats,
        class_distribution=df['Class'].value_counts().to_dict(),
        total_samples=len(df)
    )

@api_router.post("/train", response_model=TrainResponse)
async def train_model():
    """Train the ML model"""
    result = train_model_internal()
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to train model")
    
    return TrainResponse(
        success=True,
        accuracy=result['accuracy'],
        classification_report=result['report'],
        message=f"Model trained successfully with {result['accuracy']*100:.2f}% accuracy"
    )

@api_router.get("/model-status")
async def get_model_status():
    """Check if model is trained and ready"""
    global model, model_metrics
    
    if model is None:
        # Try to load existing model
        loaded = load_trained_model()
        if not loaded:
            return {
                "trained": False,
                "message": "Model not trained yet. Please train the model first."
            }
    
    return {
        "trained": True,
        "accuracy": model_metrics['accuracy'] if model_metrics else None,
        "message": "Model is ready for predictions"
    }

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_bean(input_data: PredictionInput):
    """Predict bean class"""
    global model, scaler, feature_columns, class_labels
    
    if model is None:
        loaded = load_trained_model()
        if not loaded:
            raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    # Prepare input
    input_dict = input_data.model_dump()
    input_array = np.array([[input_dict[col] for col in feature_columns]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Get class probabilities
    prob_dict = {cls: round(prob, 4) for cls, prob in zip(model.classes_, probabilities)}
    confidence = max(probabilities)
    
    return PredictionResponse(
        predicted_class=prediction,
        confidence=round(confidence, 4),
        all_probabilities=prob_dict
    )

@api_router.get("/sample-data")
async def get_sample_data():
    """Get sample data for each class"""
    df = load_dataset()
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    samples = {}
    for cls in df['Class'].unique():
        sample = df[df['Class'] == cls].iloc[0].to_dict()
        del sample['Class']
        samples[cls] = {k: round(v, 4) if isinstance(v, float) else v for k, v in sample.items()}
    
    return samples

# Include the router
app.include_router(api_router)

# Serve static HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = ROOT_DIR / 'static' / 'index.html'
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

# Static files
if (ROOT_DIR / 'static').exists():
    app.mount("/static", StaticFiles(directory=str(ROOT_DIR / 'static')), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Load model on startup if available"""
    load_trained_model()
    logger.info("Application started")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
