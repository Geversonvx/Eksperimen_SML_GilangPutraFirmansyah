import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# =============================
# SET MLFLOW DAGSHUB (WAJIB)
# =============================
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# ⚠️ JANGAN DUPLIKASI
mlflow.set_experiment("CreditCard_Default_RF_DagsHub")

# =============================
# Load data preprocessing
# =============================
X_train = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_train.csv")
X_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_test.csv")
y_train = pd.read_csv(
    "CreditCardDefaultDataset_preprocessing/y_train.csv"
).values.ravel()
y_test = pd.read_csv(
    "CreditCardDefaultDataset_preprocessing/y_test.csv"
).values.ravel()

with mlflow.start_run(run_name="RandomForest_Final_Model"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Simpan model
    os.makedirs("artifacts_extra", exist_ok=True)
    joblib.dump(model, "artifacts_extra/random_forest_final_model.joblib")

    mlflow.sklearn.log_model(model, "model")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm).to_csv(
        "artifacts_extra/confusion_matrix.csv", index=False
    )
    mlflow.log_artifact("artifacts_extra/confusion_matrix.csv")
    mlflow.log_artifact("artifacts_extra/random_forest_final_model.joblib")

    print("Training & logging ke DagsHub selesai")
