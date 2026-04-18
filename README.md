# 🚀 End-to-End MLOps Pipeline: Ethereum Fraud Detection

This repository demonstrates a complete, production-ready Machine Learning pipeline predicting fraudulent Ethereum transactions. It handles data processing, experiment tracking, model serving, and data drift monitoring.

## 🏗 Architecture Diagram
*(GitHub will automatically render this code block as a flowchart)*

```mermaid
graph LR
    A[Raw Data] --> B(Pandas Cleaning)
    B --> C{XGBoost Model}
    C -->|Track Parameters & Metrics| D[(MLflow)]
    C -->|Save Artifact| D
    D -->|Load Best Model| E[FastAPI Server]
    E -->|JSON Request| F((Prediction))
    B -->|Compare Train vs Test| G[Evidently AI]
    G -->|Generate HTML Report| H[Drift Dashboard]
