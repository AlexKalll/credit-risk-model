# src/api/pydantic_models.py
"""
Module for pydantic models
"""

from pydantic import BaseModel


class CustomerData(BaseModel):
    num__Amount: float
    num__Value: float
    num__CountryCode: float
    num__PricingStrategy: float
    num__FraudResult: float
    cat__ProductCategory_data_bundles: float
    cat__ProductCategory_financial_services: float
    cat__ProductCategory_movies: float
    cat__ProductCategory_other: float
    cat__ProductCategory_ticket: float
    cat__ProductCategory_transport: float
    cat__ProductCategory_tv: float
    cat__ProductCategory_utility_bill: float
    cat__ChannelId_ChannelId_2: float
    cat__ChannelId_ChannelId_3: float
    cat__ChannelId_ChannelId_5: float
    cat__ProviderId_ProviderId_2: float
    cat__ProviderId_ProviderId_3: float
    cat__ProviderId_ProviderId_4: float
    cat__ProviderId_ProviderId_5: float
    cat__ProviderId_ProviderId_6: float
    cat__ProductId_ProductId_10: float
    cat__ProductId_ProductId_11: float
    cat__ProductId_ProductId_12: float
    cat__ProductId_ProductId_13: float
    cat__ProductId_ProductId_14: float
    cat__ProductId_ProductId_15: float
    cat__ProductId_ProductId_16: float
    cat__ProductId_ProductId_19: float
    cat__ProductId_ProductId_2: float
    cat__ProductId_ProductId_20: float
    cat__ProductId_ProductId_21: float
    cat__ProductId_ProductId_22: float
    cat__ProductId_ProductId_23: float
    cat__ProductId_ProductId_24: float
    cat__ProductId_ProductId_27: float
    cat__ProductId_ProductId_3: float
    cat__ProductId_ProductId_4: float
    cat__ProductId_ProductId_5: float
    cat__ProductId_ProductId_6: float
    cat__ProductId_ProductId_7: float
    cat__ProductId_ProductId_8: float
    cat__ProductId_ProductId_9: float
    time__TransactionHour: int
    time__TransactionDay: int
    time__TransactionMonth: int
    time__TransactionYear: int
    agg__TotalTransactionAmount: float
    agg__AverageTransactionAmount: float
    agg__TransactionCount: float
    agg__StdTransactionAmount: float


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: bool