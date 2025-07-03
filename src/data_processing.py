# src/data_processing.py
"""
Module for data processing
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os


class TimeFeatureExtractor:
    """Class to extract time-based features from TransactionStartTime."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"])
        X["TransactionHour"] = X["TransactionStartTime"].dt.hour
        X["TransactionDay"] = X["TransactionStartTime"].dt.day
        X["TransactionMonth"] = X["TransactionStartTime"].dt.month
        X["TransactionYear"] = X["TransactionStartTime"].dt.year
        return X[
            [
                "TransactionHour",
                "TransactionDay",
                "TransactionMonth",
                "TransactionYear",
            ]
        ]

    def validate_time_range(self, X):
        """Validate that extracted time features are within expected ranges."""
        X_transformed = self.transform(X)
        if not (0 <= X_transformed["TransactionHour"].min() <= 23):
            raise ValueError("TransactionHour out of range [0, 23]")
        if not (1 <= X_transformed["TransactionDay"].min() <= 31):
            raise ValueError("TransactionDay out of range [1, 31]")
        return True

    def get_feature_names_out(self, input_features=None):
        return [
            "TransactionHour",
            "TransactionDay",
            "TransactionMonth",
            "TransactionYear",
        ]


class AggregateFeatureExtractor:
    """Class to extract aggregate features per CustomerId."""

    def __init__(self):
        self.groups = None

    def fit(self, X, y=None):
        self.groups = X.groupby("CustomerId")
        return self

    def transform(self, X):
        X = X.copy()
        agg_features = (
            self.groups["Amount"]
            .agg(["sum", "mean", "count", "std"])
            .rename(
                columns={
                    "sum": "TotalTransactionAmount",
                    "mean": "AverageTransactionAmount",
                    "count": "TransactionCount",
                    "std": "StdTransactionAmount",
                }
            )
        )
        return X.merge(agg_features, on="CustomerId", how="left")[
            [
                "TotalTransactionAmount",
                "AverageTransactionAmount",
                "TransactionCount",
                "StdTransactionAmount",
            ]
        ]

    def get_feature_names_out(self, input_features=None):
        return [
            "TotalTransactionAmount",
            "AverageTransactionAmount",
            "TransactionCount",
            "StdTransactionAmount",
        ]


class DataProcessor:
    """Class to manage data processing pipeline and execution."""

    def __init__(
        self,
        raw_path="data/raw/data.csv",
        processed_path="data/processed/processed_data.csv",
    ):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.pipeline = self._create_pipeline()

    def _load_data(self):
        return pd.read_csv(self.raw_path)

    def _create_pipeline(self):
        categorical_cols = [
            "ProductCategory",
            "ChannelId",
            "ProviderId",
            "ProductId",
            "CurrencyCode",
        ]
        numerical_cols = [
            "Amount",
            "Value",
            "CountryCode",
            "PricingStrategy",
            "FraudResult",
        ]
        time_col = ["TransactionStartTime"]
        agg_cols = ["CustomerId", "Amount"]

        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
            ]
        )

        time_transformer = Pipeline(
            steps=[("time_extractor", TimeFeatureExtractor())]
        )

        agg_transformer = Pipeline(
            steps=[("agg_extractor", AggregateFeatureExtractor())]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
                ("time", time_transformer, time_col),
                ("agg", agg_transformer, agg_cols),
            ],
            remainder="passthrough",
        )

        return Pipeline(steps=[("preprocessor", preprocessor)])

    def process(self):
        df = self._load_data()
        processed_data = self.pipeline.fit_transform(df)
        n_features = processed_data.shape[1]
        column_names = self.pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        if len(column_names) != n_features:
            raise ValueError(
                f"Feature name count ({len(column_names)}) does not match "
                f"transformed data shape ({n_features})"
            )

        processed_df = pd.DataFrame(processed_data, columns=column_names)
        processed_df.index = df.index

        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        processed_df.to_csv(self.processed_path, index=False)
        print(f"Processed data saved to {self.processed_path}")
        print(f"Number of features: {n_features}")
        print(f"Column names: {list(processed_df.columns)}")


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process()