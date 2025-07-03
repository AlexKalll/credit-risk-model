# src/api/main.py
"""
Module for main
"""

from fastapi import FastAPI
import mlflow.sklearn
import numpy as np
from .pydantic_models import CustomerData, PredictionResponse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
logger.info("MLflow tracking URI set to http://127.0.0.1:5000")

# Load the best model from MLflow as a
# scikit-learn model (version 3 of Credit_Risk_Model)
try:
    model = mlflow.sklearn.load_model("models:/Credit_Risk_Model/4")
    logger.info(
        "Model loaded from MLflow Registry "
        "(version 4) as scikit-learn model"
    )
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


@app.get("/")
async def read_root():
    return {"message": "Credit Risk Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer_data: CustomerData):
    # Convert Pydantic model to numpy array
    data = np.array(
        [
            [
                getattr(customer_data, field)
                for field in customer_data.__fields__.keys()
            ]
        ]
    )
    # Get prediction probability
    try:
        probability = model.predict_proba(data)[0][
            1
        ]  # Probability of high risk (class 1)
        is_high_risk = probability > 0.5  # Threshold for binary classification
        logger.info(
            "Prediction made: "
            f"probability={probability}, is_high_risk={is_high_risk}"
        )
        return PredictionResponse(
            risk_probability=probability, is_high_risk=is_high_risk
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise