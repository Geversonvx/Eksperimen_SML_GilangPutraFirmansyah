import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
dagshub.init(
    repo_owner="Geversonvx",
    repo_name="Eksperimen_SML_GilangPutraFirmansyah",
    mlflow=True,
)

def main():
    # =========================
    # Load dataset hasil preprocessing
    # =========================
    X_train = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_train.csv")
    X_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_test.csv")
    y_train = pd.read_csv(
        "CreditCardDefaultDataset_preprocessing/y_train.csv"
    ).values.ravel()
    y_test = pd.read_csv(
        "CreditCardDefaultDataset_preprocessing/y_test.csv"
    ).values.ravel()

    # =========================
    # Aktifkan MLflow autolog
    # =========================
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Credit_Card_Default_Basic_Model")

    mlflow.sklearn.autolog()

    # =========================
    # Training
    # =========================
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)

        # =========================
        # Evaluasi
        # =========================
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy :", acc)
        print("Precision:", prec)
        print("Recall   :", rec)
        print("F1-score :", f1)


if __name__ == "__main__":
    main()
