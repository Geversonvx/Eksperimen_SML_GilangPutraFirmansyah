import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(input_path, output_dir, test_size=0.2, random_state=42):
    df = pd.read_excel(input_path, skiprows=1)

    df.drop(columns=["ID"], inplace=True)

    df.rename(
        columns={"default payment next month": "default.payment.next.month"},
        inplace=True
    )

    target_col = "default.payment.next.month"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train, columns=X.columns).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    joblib.dump(imputer, f"{output_dir}/imputer.joblib")
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")

    print("Preprocessing selesai. Dataset siap dilatih.")


if __name__ == "__main__":
    preprocess_data(
        "CreditCardDefaultDataset_raw/default of credit card clients.xls",
        "CreditCardDefaultDataset_preprocessing"
    )
