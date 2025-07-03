# src/target_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import pytz
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load raw data for RFM calculation
raw_path = "data/raw/data.csv"
try:
    raw_df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw data from {raw_path}")
except FileNotFoundError:
    logger.error(f"Raw data file not found at {raw_path}")
    raise

# Load processed data from Task 3
processed_path = "data/processed/processed_data.csv"
try:
    df = pd.read_csv(processed_path)
    logger.info(f"Loaded processed data from {processed_path}")
except FileNotFoundError:
    logger.error(f"Processed data file not found at {processed_path}")
    raise

# Define snapshot date (today: June 30, 2025)
snapshot_date = datetime(2025, 6, 30, tzinfo=pytz.UTC)

# Convert TransactionStartTime to datetime in raw data
raw_df["TransactionStartTime"] = pd.to_datetime(raw_df["TransactionStartTime"])

# Calculate RFM metrics per CustomerId from raw data
rfm_data = (
    raw_df.groupby("CustomerId")
    .agg(
        {
            "TransactionStartTime": lambda x: (
                snapshot_date - x.max()
            ).days,  # Recency
            "TransactionId": "count",  # Frequency
            "Amount": "sum",  # Monetary
        }
    )
    .rename(
        columns={
            "TransactionStartTime": "Recency",
            "TransactionId": "Frequency",
            "Amount": "Monetary",
        }
    )
)

# Handle potential missing values or zeros in Frequency and Monetary
rfm_data["Frequency"] = rfm_data["Frequency"].replace(
    0, 1
)  # Avoid division by zero
rfm_data["Monetary"] = rfm_data["Monetary"].replace(
    0, 0.01
)  # Small non-zero value
percentile_99 = rfm_data["Monetary"].quantile(0.99)
rfm_data["Monetary"] = rfm_data["Monetary"].clip(
    lower=-percentile_99, upper=percentile_99
)  # cap or remove extreme Monetary values

# Pre-process RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)

# Cluster customers using K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(rfm_scaled)

# Add cluster labels to RFM data
rfm_data["Cluster"] = clusters

# Analyze clusters to identify high-risk group
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=["Recency", "Frequency", "Monetary"],
)
cluster_sizes = rfm_data["Cluster"].value_counts()

# Custom metric for high-risk: inverse of (Frequency + |Monetary|)
# to prioritize low engagement
engagement_score = 1 / (
    cluster_centers["Frequency"] + cluster_centers["Monetary"].abs()
)
high_risk_cluster = (
    engagement_score.idxmax()
)  # Highest score indicates lowest engagement

# Create is_high_risk column
rfm_data["is_high_risk"] = (rfm_data["Cluster"] == high_risk_cluster).astype(
    int
)

# Merge is_high_risk to raw data on CustomerId
raw_df = raw_df.merge(rfm_data[["is_high_risk"]], on="CustomerId", how="left")

# Select TransactionId and is_high_risk
target_data = raw_df[["TransactionId", "is_high_risk"]]

# Merge to processed data on remainder__TransactionId
df = df.merge(
    target_data,
    left_on="remainder__TransactionId",
    right_on="TransactionId",
    how="left",
)
df = df.drop(columns=["TransactionId"])  # Remove duplicate column

# Handle any missing is_high_risk values
if df["is_high_risk"].isna().any():
    logger.warning("Some transactions missing is_high_risk; filled with 0")
    df["is_high_risk"] = df["is_high_risk"].fillna(0)

# Save updated dataset
output_path = "data/processed/processed_data_with_target.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
logger.info(f"Updated dataset saved to {output_path}")

# Optional: Validate cluster assignment
logger.info("Cluster Centers (unscaled):")
print(cluster_centers)
logger.info("\nCluster Sizes:")
print(cluster_sizes)
logger.info(f"\nHigh-Risk Cluster: {high_risk_cluster} (Lowest Engagement)")
