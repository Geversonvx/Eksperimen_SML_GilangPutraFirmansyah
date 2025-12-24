import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Load data
X_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_test.csv")
y_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/y_test.csv")

# Load model final
model = joblib.load("random_forest_final_model.joblib")

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Hitung metrik
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# MLflow logging
mlflow.set_experiment("CreditCardDefault_Evaluation")

with mlflow.start_run(run_name="RandomForest_Final_Eval"):
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        model, artifact_path="model", registered_model_name="CreditCardDefault_RF"
    )

print("Evaluasi selesai & tercatat di MLflow")
