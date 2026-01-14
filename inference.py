import uvicorn
import joblib
import os
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional
from sklearn.preprocessing import PowerTransformer



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CHOSEN_THRESHOLD = 0.6

# FastAPI app instance
app = FastAPI(title="Stroke Prediction API", version="1.0.0")

# Pydantic models for request/response
class StrokeInput(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age in years")
    gender: str  # Male/Female
    height: float = Field(..., ge=50, le=250, description="Height in centimeters (cm)")
    weight: float = Field(..., ge=10, le=500, description="Weight in kilograms (kg)")
    systolic_bp: float = Field(..., ge=50, le=250, description="Systolic Blood Pressure")
    diastolic_bp: float = Field(..., ge=30, le=150, description="Diastolic Blood Pressure")
    BMI: float = None

class StrokePrediction(BaseModel):
    prediction: int  # 0 or 1
    probability: float
    risk_level: str

class HealthCheck(BaseModel):
    status: Optional[str] = "healthy"
    model_loaded: bool = False
    version: Optional[str] = "1.0.0"
    message: Optional[str] = None

# Global variables
model = None

def load_model():
    """Load the trained model and preprocessor"""
    global model
    try:
        # Get the current directory (works both locally and on Render)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load model and preprocessor from models folder
        model_path = os.path.join(current_dir, 'models', 'SVC_model.joblib')
        
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False



@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Stroke Prediction API...")
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return HealthCheck(
        message="Stroke Prediction API is running",
        status="healthy"
        )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    
    return HealthCheck(
        status="healthy",
        model_loaded=True if model is not None else False,
        version="1.0.0"
    )

@app.post("/predict", response_model=StrokePrediction)
async def predict_stroke(input_data: StrokeInput):
    """
    Asynchronously predict stroke risk from a single patient's input features.
    This function:
    - Validates that the global `model` is loaded.
    - Builds a one-row pandas DataFrame from `input_data`.
    - Performs feature engineering and basic outlier clipping:
        - Applies Yeo-Johnson power transforms to height, weight, BMI (and intended for DBP).
        - Clips systolic and diastolic blood pressure to the 1st–99th percentile range.
        - Computes BMI from height (cm) and weight (kg), clips BMI to [10, 80], then transforms it.
    - Drops raw columns used to compute engineered features and runs inference:
        - `prediction = model.predict(processed_data)[0]`
        - `probability = model.predict_proba(processed_data)[0][1]`
    Risk level thresholding:
    - "High" risk if `probability >= CHOSEN_THRESHOLD`
    - "Low" risk if `probability < CHOSEN_THRESHOLD`
    `CHOSEN_THRESHOLD` should be a float in [0.0, 1.0] (commonly 0.5 by default, but it may be
    tuned for your desired sensitivity/specificity trade-off).
    Args:
            input_data (StrokeInput): Patient input payload (age, gender, height, weight,
                    systolic_bp, diastolic_bp).
    Returns:
            StrokePrediction: Object containing:
                    - prediction (int): Model class prediction (e.g., 0/1).
                    - probability (float): Predicted probability for the positive class.
                    - risk_level (str): "High" or "Low" based on `CHOSEN_THRESHOLD`.
    Raises:
            HTTPException: 503 if the model or preprocessor is not loaded.
            HTTPException: 500 if preprocessing or prediction fails for any reason.
    """
 
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert input to pandas DataFrame
        df = pd.DataFrame({
            "age": [input_data.age],
            "gender": [input_data.gender],
            "height": [input_data.height],
            "weight": [input_data.weight],
            "Systolic_BP": [input_data.systolic_bp],
            "Diastolic_BP": [input_data.diastolic_bp],
            "BMI":  [input_data.BMI],
        })

        # Data preprocessing steps
        df['height_yj'] = PowerTransformer(method="yeo-johnson").fit_transform(df[['height']])

        # Weight.
        df['weight_yj'] = PowerTransformer(method="yeo-johnson").fit_transform(df[['weight']])

        # Systolic Blood Pressure.
        df['Systolic_BP'] = df['Systolic_BP'].clip(df['Systolic_BP'].quantile(0.01),
                                                df['Systolic_BP'].quantile(0.99))

        # Diastolic BP.
        df['Diastolic_BP'] = df['Diastolic_BP'].clip(df['Diastolic_BP'].quantile(0.01),
                                                    df['Diastolic_BP'].quantile(0.99))
        df['DBP_yj'] = PowerTransformer(method='yeo-johnson').fit_transform(df[['Diastolic_BP']])
        df["gender"] = [1 if gender == "Male" else 0 for gender in df['gender']]


        # Body Mass Index (BMI).
        df['BMI'] = df['weight'] / ((df['height']/100)**2) # numerical adjustments.
        df['BMI'] = df['BMI'].clip(lower=10, upper=80)
        df['BMI_yj'] = PowerTransformer(method="yeo-johnson").fit_transform(df[['BMI']])

        # Preprocess the data
        processed_data = df.drop(columns=["BMI", "Diastolic_BP", "height", "weight"], axis=1)

        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Determine risk level
        risk_level = "High" if probability >= CHOSEN_THRESHOLD else "Low" 
        
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
