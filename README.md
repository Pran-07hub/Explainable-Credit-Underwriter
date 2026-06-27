# Explainable Credit Underwriter

An end-to-end AI-powered credit underwriting platform that combines machine learning, explainable AI, and generative AI to automate credit risk assessment and produce audit-ready underwriting memos.

## Overview

Traditional credit scoring systems often act as black boxes. This project addresses that challenge by integrating:

- XGBoost for default risk prediction
- SHAP for customer-level explainability
- Generative AI for automated underwriting reports
- Streamlit for interactive decision support

---

## Architecture

Customer Financial Profile
↓
XGBoost Risk Model
↓
Default Probability
↓
SHAP Explainability Engine
↓
Top Risk Drivers
↓
LLM Credit Review Memo

---

## Features

### Credit Risk Modeling

- Home Credit Default Risk Dataset
- XGBoost Classifier
- ROC-AUC: 0.75
- Test Accuracy: 0.76

### Explainable AI

- Global SHAP Feature Importance
- SHAP Beeswarm Summary Plot
- Individual Applicant Waterfall Explanations

### GenAI Layer

Generates professional underwriting memos using:

- Applicant attributes
- Predicted risk score
- SHAP-derived risk drivers
- Feature descriptions

### Interactive Dashboard

- Real-time applicant evaluation
- Risk scoring
- Explainability visualization
- Automated decision memo generation

---

## Results

| Metric | Value |
|----------|----------|
| ROC-AUC | 0.75 |
| Test Accuracy | 0.76 |
| Positive Class Recall | 0.58 |
| Positive Class Precision | 0.18 |

---

## Sample Explainability

### Global Feature Importance

![SHAP Importance](Assets/shap_feature_importance.png)

### SHAP Summary Plot

![SHAP Summary](Assets/shap_summary.png)

### Customer Waterfall Plot

![Waterfall](Assets/waterfall_plot.png)

### Customer Memo

CREDIT REVIEW MEMO

Risk Assessment:
The applicant's default risk score of 15.02% is driven by their relatively high outstanding goods price, prolonged employment period, and older age. However, these risk factors are somewhat mitigated by their strong credit history, good education level, and consistent payment behavior. Despite these concerns, the overall creditworthiness of the applicant remains solid.

Recommendation:
Approve, as the risk score falls below the threshold of 18%.
---

## Tech Stack

- Python
- Pandas
- NumPy
- XGBoost
- SHAP
- Scikit-Learn
- Streamlit
- Hugging Face Transformers
