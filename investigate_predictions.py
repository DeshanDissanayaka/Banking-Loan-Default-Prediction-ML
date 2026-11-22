
import sqlite3
import json
import joblib
import pandas as pd

from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "artifacts" / "models"
DATABASE_PATH = ROOT / "artifacts" / "loan_predictions.db"

NUMERICAL_FEATURES = [
    "age",
    "employment_years",
    "annual_income_lkr",
    "monthly_income_lkr",
    "existing_loans",
    "outstanding_debt_lkr",
    "credit_score",
    "credit_history_years",
    "previous_defaults",
    "delinquencies_last_2yrs",
    "credit_utilization",
    "loan_amount_lkr",
    "loan_term_months",
    "interest_rate_pct",
    "monthly_emi_lkr",
    "debt_to_income_ratio",
    "collateral_value_lkr"
]


# ==========================================
# 2. LOAD SAVED PREPROCESSING OBJECTS
# ==========================================

scaler = joblib.load(
    MODEL_DIR / "standard_scaler.pkl"
)

numerical_imputer = joblib.load(
    MODEL_DIR / "numerical_imputer.pkl"
)

model = joblib.load(
    MODEL_DIR / "final_model.pkl"
)


print("\n========== SCALER INFORMATION ==========")

for feature, mean, scale in zip(
    scaler.feature_names_in_,
    scaler.mean_,
    scaler.scale_
):
    print(
        f"{feature}: "
        f"Mean={mean:.6f}, "
        f"Scale={scale:.6f}"
    )


# ==========================================
# 3. LOAD SAVED APPLICATIONS
# ==========================================

connection = sqlite3.connect(DATABASE_PATH)

records = connection.execute("""
    SELECT id, customer_inputs
    FROM prediction_history
    ORDER BY id DESC
""").fetchall()

connection.close()


# ==========================================
# 4. CHECK ORIGINAL INPUT VALUES
# ==========================================

print("\n========== INPUT AND SCALING CHECK ==========")

for record_id, customer_inputs in records:

    inputs = json.loads(customer_inputs)

    print(f"\nRECORD ID: {record_id}")

    for feature in NUMERICAL_FEATURES:

        raw_value = pd.to_numeric(
            inputs.get(feature),
            errors="coerce"
        )

        if pd.isna(raw_value):
            print(f"{feature}: Missing")
            continue

        feature_index = list(
            scaler.feature_names_in_
        ).index(feature)

        mean = scaler.mean_[feature_index]
        scale = scaler.scale_[feature_index]

        standardized_value = (
            raw_value - mean
        ) / scale

        print(
            f"{feature}: "
            f"Input={raw_value:.4f}, "
            f"Training Mean={mean:.4f}, "
            f"Training Scale={scale:.4f}, "
            f"Scaled={standardized_value:.4f}"
        )


# ==========================================
# 5. CHECK MODEL PROBABILITY
# ==========================================

print("\n========== MODEL PROBABILITY CHECK ==========")

for record_id, customer_inputs in records:

    inputs = json.loads(customer_inputs)

    row = pd.DataFrame([inputs])

    row["application_date"] = pd.to_datetime(
        row["application_date"]
    )

    row["application_year"] = row["application_date"].dt.year
    row["application_month"] = row["application_date"].dt.month
    row["application_day"] = row["application_date"].dt.day
    row["application_dayofweek"] = row["application_date"].dt.dayofweek

    row.drop(columns=["application_date"], inplace=True)

    numerical_columns = list(
        numerical_imputer.feature_names_in_
    )

    numerical_data = pd.DataFrame(
        numerical_imputer.transform(
            row[numerical_columns]
        ),
        columns=numerical_columns
    )

    categorical_columns = list(
        joblib.load(
            MODEL_DIR / "categorical_imputer.pkl"
        ).feature_names_in_
    )

    categorical_imputer = joblib.load(
        MODEL_DIR / "categorical_imputer.pkl"
    )

    categorical_data = pd.DataFrame(
        categorical_imputer.transform(
            row[categorical_columns]
        ),
        columns=categorical_columns
    )

    encoder = joblib.load(
        MODEL_DIR / "onehot_encoder.pkl"
    )

    encoded = encoder.transform(categorical_data)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )

    processed = pd.concat(
        [numerical_data, encoded_df],
        axis=1
    )

    processed[numerical_columns] = scaler.transform(
        processed[numerical_columns]
    )

    processed = processed[model.feature_names_in_]

    raw_score = model.decision_function(processed)[0]

    probability = model.predict_proba(processed)[0, 1]

    print(f"\nRecord {record_id}")
    print(f"Decision score: {raw_score:.6f}")
    print(f"Default probability: {probability * 100:.6f}%")


print("\n========== DIAGNOSTIC COMPLETED ==========")
