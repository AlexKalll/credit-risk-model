# Credit Risk Probability Model for Alternative Data

## Project Overview
This project aims to develop an end-to-end credit risk probability model for Bati Bank, leveraging transaction data from an e-commerce platform. The model will assess customer creditworthiness to facilitate a "buy-now-pay-later" service, transforming behavioral data into a predictive risk signal by analyzing customer Recency, Frequency, and Monetary (RFM) patterns.

## Dataset Schema
The dataset contains transaction-level information with the following fields:
- `TransactionId`: Unique transaction identifier
- `BatchId`: ID for a batch of transactions
- `AccountId`: Unique customer ID
- `SubscriptionId`: Customer’s subscription ID
- `CustomerId`: Linked to Account
- `CurrencyCode`: Currency used
- `CountryCode`: Numeric country code
- `ProviderId`: Source of purchased item
- `ProductId`: Purchased item
- `ProductCategory`: Category of item
- `ChannelId`: Platform used (web, Android, iOS, etc.)
- `Amount`: Transaction value (positive = debit, negative = credit)
- `Value`: Absolute amount (absolute amount of the transaction [cite: 3])
- `TransactionStartTime`: When transaction began
- `PricingStrategy`: Pricing category used
- `FraudResult`: Fraud status of transaction (1 = fraud, 0 = no fraud)
---
## Credit Scoring Business Understanding

#### 1. Basel II’s Emphasis on Risk Measurement: Need for Interpretability and Documentation 🧩
Basel II introduced risk-sensitive regulatory capital under **Pillar 1**, requiring financial institutions to hold capital proportional to the **Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)**[cite: 4]. This regulatory environment demands:

- **Transparent model logic** so regulators and internal validators can trace how each feature influences PD.
- **Thorough documentation** of data sources, transformations, assumptions, and performance metrics to satisfy auditing standards and the three lines of defense[cite: 5].

An interpretable, well-documented model ensures compliance, better risk governance, and trust in capital adequacy.

#### 2. Necessity of Proxy Variables for "Default" Labels & Associated Business Risks ⚠️

- **Why proxy variables?** True default events are rare and key datasets seldom include explicit “default” flags. To train models, we must use **proxy events** (e.g., 60+ days past due, charge-offs) as labels[cite: 6].

- **Business risks of proxy-based predictions:** 1. **Label noise** – proxies may not consistently indicate eventual default, leading to inaccurate PD estimates. 
  2. **Regulatory judgment** – if proxy misaligns with regulatory definitions, model outcomes could be challenged in audits. 
  3. **Decision bias** – incorrectly flagging or clearing borrowers based on imperfect signals could result in credit losses or missed business opportunities.


#### 3. Trade-Offs: Simple vs. Complex Models in Regulated Financial Context 🧠

| Aspect              | Simple, Interpretable Models (e.g., Logistic Regression + WoE) | Complex, High-Performance Models (e.g., Gradient Boosting) |
|---------------------|---------------------------------------------------------------|------------------------------------------------------------|
| **Interpretability** | ✅ Clear feature effects, easy rationale for credit decisions | ❌ Often opaque; needs auxiliary explainability tools       |
| **Regulatory Approval** | ✅ Easily documented, validated, and approved               | ⚠️ Harder to validate; may face resistance                 |
| **Performance** | Moderate predictive power                                     | [cite_start]✅ Higher discrimination, especially with non-linear relationships [cite: 7] |
| **Implementation Cost** | Low complexity, fast deployment                               | Higher computational cost and integration overhead         |
| **Governance** | Easier to monitor post-launch                                 | Requires advanced governance for drift, fairness, feature updating |

**In a regulated setting**, the preference often leans toward a simpler, interpretable model unless a complex model’s performance gain is substantial and justifiable in terms of:

- enhanced risk-adjusted capital efficiency,
- investment into documentation (e.g., SHAP values, surrogate models),
- and ongoing validation frameworks.

---

#### 🧾 Summary

To comply with Basel II:

- **Interpretability and strong documentation** are non-negotiable for demonstrating accurate risk measurement and governance.
- **Proxy variables** are essential for modeling-but come with label risks that must be managed.
- **Model choice** is a balancing act: simplicity ensures regulatory alignment; complexity can deliver superior performance—but demands robust validation, documentation, and justification.

## Project Structure
This repository follows a standardized project structure for developing, deploying, and automating a credit risk probability model.

```
credit-risk-model/                      # Root project folder
    ├── .github/                            # GitHub CI/CD workflows
    │   └── workflows/
    │       └── ci.yml                      # GitHub Actions config
    │
    ├── data/                               # Data storage (.gitignored)
    │   ├── raw/                            # Original raw data files
    │   └── processed/                      # Processed/feature-engineered data
    │
    ├── notebooks/                          # Exploratory analysis
    │   ├── 01_data_exploration.ipynb       # Initial EDA and data profiling
    │   ├── 02_feature_engineering.ipynb    # Feature analysis and experiments
    │   ├── 03_model_experimentation.ipynb  # Model training and evaluation
    │   ├── 04_results_analysis.ipynb       # Final metrics and visualizations
    │   └── README.md                       # Notebook documentation
    │
    ├── src/                                # Production code
    │   ├── __init__.py                     # Python package marker
    │   ├── data_processing.py              # Feature engineering pipeline
    │   ├── train.py                        # Model training script
    │   ├── predict.py                      # Inference script
    │   │
    │   └── api/                            # Deployment components
    │       ├── __init__.py
    │       ├── main.py                     # FastAPI application
    │       └── pydantic_models.py          # API request/response schemas
    │
    ├── tests/                              # Unit tests
    │   ├── __init__.py
    │   ├── test_data_processing.py         # Feature engineering tests
    │   └── test_models.py                  # Model validation tests
    │
    ├── Dockerfile                          # Container configuration
    ├── docker-compose.yml                  # Multi-container setup
    ├── requirements.txt                    # Python dependencies
    ├── .gitignore                          # Ignored files/directories
    └── README.md                           # Project documentation
```