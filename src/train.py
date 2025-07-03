# train.py
"""
Module for train
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import logging
from mlflow.pyfunc import PythonModel, PythonModelContext

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define PythonModel for PyFunc
class SklearnModelWrapper(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(
        self, context: PythonModelContext, model_input: np.ndarray
    ) -> np.ndarray:
        return self.model.predict(model_input)

    def predict_proba(
        self, context: PythonModelContext, model_input: np.ndarray
    ) -> np.ndarray:
        return self.model.predict_proba(model_input)


# Load data
data_path = "data/processed/processed_data_with_target.csv"
try:
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data from {data_path}")
except FileNotFoundError:
    logger.error(f"Data file not found at {data_path}")
    raise

# Impute NaN values in agg__StdTransactionAmount with 0
df["agg__StdTransactionAmount"] = df["agg__StdTransactionAmount"].fillna(0)
logger.info("Imputed NaN values in agg__StdTransactionAmount with 0")

# Split features and target, dropping non-numeric columns
X = df.drop(
    columns=[
        "is_high_risk",
        "remainder__TransactionId",
        "remainder__BatchId",
        "remainder__AccountId",
        "remainder__SubscriptionId",
    ]
)
y = df["is_high_risk"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
logger.info(f"Data split: {X_train.shape} train, {X_test.shape} test")

# Initialize MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
mlflow.set_experiment("Credit_Risk_Modeling")

# Store run IDs
lr_run_id = None
rf_run_id = None

# Model 1: Logistic Regression
with mlflow.start_run(run_name="Logistic_Regression") as run:
    logger.info("Training Logistic Regression")
    lr = LogisticRegression(random_state=42, max_iter=2000)
    param_grid_lr = {"C": [0.01, 0.1, 1.0, 10.0]}
    grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring="f1")
    grid_search_lr.fit(X_train, y_train)
    best_lr = grid_search_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)

    # Evaluate
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr),
        "roc_auc": roc_auc_score(y_test, best_lr.predict_proba(X_test)[:, 1]),
    }
    logger.info(f"Logistic Regression Metrics: {metrics}")
    mlflow.log_params(grid_search_lr.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        best_lr,
        "model",
        input_example=X_train[:1],
        signature=mlflow.models.signature.infer_signature(X_train, y_train),
    )
    lr_run_id = run.info.run_id

# Model 2: Random Forest
with mlflow.start_run(run_name="Random_Forest") as run:
    logger.info("Training Random Forest")
    rf = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="f1")
    grid_search_rf.fit(X_train, y_train)
    best_rf = grid_search_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    # Evaluate
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf),
        "roc_auc": roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]),
    }
    logger.info(f"Random Forest Metrics: {metrics}")
    mlflow.log_params(grid_search_rf.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        best_rf,
        "model",
        input_example=X_train[:1],
        signature=mlflow.models.signature.infer_signature(X_train, y_train),
    )
    mlflow.pyfunc.log_model(
        "pyfunc_model",
        python_model=SklearnModelWrapper(best_rf),
        input_example=X_train[:1],
        signature=mlflow.models.signature.infer_signature(X_train, y_train),
    )
    rf_run_id = run.info.run_id

# Register the best model (compare F1 score)
if grid_search_lr.best_score_ > grid_search_rf.best_score_:
    best_model = best_lr
    best_run_id = lr_run_id
else:
    best_model = best_rf
    best_run_id = rf_run_id

with mlflow.start_run(run_id=best_run_id):
    mlflow.sklearn.log_model(
        best_model,
        "best_model",
        input_example=X_train[:1],
        signature=mlflow.models.signature.infer_signature(X_train, y_train),
    )
    model_info = mlflow.register_model(
        f"runs:/{best_run_id}/best_model", "Credit_Risk_Model"
    )
    logger.info(f"Registered model version {model_info.version}")

logger.info("Model training and registration completed.")