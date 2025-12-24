import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

# ===============================
# Load dataset hasil preprocessing
# ===============================
X_train = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_train.csv")
X_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_test.csv")
y_train = pd.read_csv(
    "CreditCardDefaultDataset_preprocessing/y_train.csv"
).values.ravel()
y_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/y_test.csv").values.ravel()

# ===============================
# Setup MLflow
# ===============================
mlflow.set_experiment("Credit Card Default - Hyperparameter Tuning")

# ===============================
# Model & Hyperparameter
# ===============================
model = LogisticRegression(max_iter=1000)

param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}

grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)

# ===============================
# Training + Tracking
# ===============================
with mlflow.start_run():
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log ke MLflow
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(best_model, "model")

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
    print("F1-score:", f1)
