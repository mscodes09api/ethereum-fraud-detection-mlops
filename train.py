import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

def generate_synthetic_data(n_samples=1000):
    """Generates synthetic Ethereum transaction data for demonstration."""
    np.random.seed(42)
    data = {
        'Avg_min_between_sent_tnx': np.random.uniform(0, 1000, n_samples),
        'Avg_min_between_received_tnx': np.random.uniform(0, 1000, n_samples),
        'Time_Diff_between_first_and_last_Mins': np.random.uniform(0, 10000, n_samples),
        'Sent_tnx': np.random.randint(0, 100, n_samples),
        'Received_Tnx': np.random.randint(0, 100, n_samples),
        'Number_of_Created_Contracts': np.random.randint(0, 10, n_samples),
    }
    df = pd.DataFrame(data)
    # Simple logic for fraud label: if sent_tnx is high and created contracts is high
    df['is_fraud'] = ((df['Sent_tnx'] > 80) & (df['Number_of_Created_Contracts'] > 5)).astype(int)
    return df

def train_model():
    # 1. Load/Generate Data
    df = generate_synthetic_data()
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Start MLflow Run
    with mlflow.start_run() as run:
        params = {
            "objective": "binary:logistic",
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 100
        }
        mlflow.log_params(params)
        
        # 3. Train Model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # 4. Evaluate
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # 5. Log Model
        mlflow.xgboost.log_model(model, "xgb_model")
        
        print(f"Model trained. Accuracy: {accuracy:.4f}")
        print(f"MLFLOW_RUN_ID: {run.info.run_id}")
        
        # Write Run ID to a file for the API to use if needed
        with open(".mlflow_run_id", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    train_model()
