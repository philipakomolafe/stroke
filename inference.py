from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(title="Stroke Prediction API", version="1.0.0")

# Pydantic models for request/response
class StrokeInput(BaseModel):
    age: float
    gender: str  # Male/Female
    height: float  # in cm
    weight: float  # in kg
    systolic_bp: float  # Systolic Blood Pressure
    diastolic_bp: float  # Diastolic Blood Pressure
    bmi: float

class StrokePrediction(BaseModel):
    prediction: int  # 0 or 1
    probability: float
    risk_level: str

# Global variables
model = None
preprocessor = None

def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    global model, preprocessor
    try:
        # Get the current directory (works both locally and on Render)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load model and preprocessor from models folder
        model_path = os.path.join(current_dir, 'models', 'RandomForest_best_model.joblib')
        preprocessor_path = os.path.join(current_dir, 'models', 'preprocessor.joblib')
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Model and preprocessor loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model/preprocessor: {e}")
        return False



@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Stroke Prediction API...")
    load_model_and_preprocessor()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Stroke Prediction API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=StrokePrediction)
async def predict_stroke(input_data: StrokeInput):
    """Make stroke prediction"""
    try:
        if model is None or preprocessor is None:
            raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
        
        # Convert input to array
        input_array = np.array([[
            input_data.age,
            input_data.gender,
            input_data.height,
            input_data.weight,
            input_data.systolic_bp,
            input_data.diastolic_bp,
            input_data.bmi
        ]])
        
        # Preprocess the data
        processed_data = preprocessor.transform(input_array)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Determine risk level
        risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
        
        return StrokePrediction(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def begin_inference():
    """Initialize the inference process"""
    print("Beginning inference process...")
    logger.info("Inference process started")
    
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run(
        "inference:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    begin_inference()
    print("INFERENCE COMPLETED")