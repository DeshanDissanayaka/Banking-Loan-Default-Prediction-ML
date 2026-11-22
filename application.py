
from flask import Flask, render_template, request
import pandas as pd
import sqlite3
import json

from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from src.pipeline.predict_pipeline import LoanPredictionPipeline


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

DATABASE_PATH = (
    PROJECT_ROOT / "artifacts" / "loan_predictions.db"
)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

prediction_pipeline = LoanPredictionPipeline()


# ==========================================
# 2. MODEL INPUT FEATURES
# ==========================================

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

CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "education_level",
    "employment_type",
    "home_ownership",
    "loan_purpose",
    "has_cosigner",
    "collateral_type"
]


# ==========================================
# 3. DATABASE INITIALIZATION
# ==========================================

def initialize_database():

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_date TEXT,
            customer_inputs TEXT,
            prediction TEXT,
            default_probability REAL,
            created_at TEXT
        )
    """)

    connection.commit()
    connection.close()


# ==========================================
# 4. SAVE PREDICTION TO DATABASE
# ==========================================

def save_prediction(
    application_date,
    customer_inputs,
    prediction
):

    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute("""
        INSERT INTO prediction_history (
            application_date,
            customer_inputs,
            prediction,
            default_probability,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        application_date,
        json.dumps(customer_inputs),
        prediction["result"],
        prediction["default_probability"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


# ==========================================
# 5. MODEL PERFORMANCE EVALUATION
# ==========================================

def evaluate_model():

    # Check test dataset availability
    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():

        app.logger.warning(
            "Test dataset files were not found."
        )

        return None

    # Load processed test features
    X_test = pd.read_csv(X_TEST_PATH)

    # Load actual test labels
    y_test_df = pd.read_csv(Y_TEST_PATH)

    # Identify target column
    possible_target_columns = [
        "default_flag",
        "target",
        "label"
    ]

    target_column = next(
        (
            column
            for column in possible_target_columns
            if column in y_test_df.columns
        ),
        None
    )

    if target_column is not None:

        y_test = y_test_df[target_column]

    else:

        # Ignore an exported pandas index if present
        valid_columns = [
            column
            for column in y_test_df.columns
            if not str(column).startswith("Unnamed:")
        ]

        if len(valid_columns) != 1:

            raise ValueError(
                "Unable to identify the target column in y_test.csv."
            )

        y_test = y_test_df[valid_columns[0]]

    # Match the model's original feature order
    model = prediction_pipeline.model

    X_test = X_test[
        model.feature_names_in_
    ]

    # Convert target labels to numeric if necessary
    if y_test.dtype == "object":

        y_test = y_test.astype(str).str.strip().str.lower()

        y_test = y_test.map({
            "non-default": 0,
            "non_default": 0,
            "no": 0,
            "0": 0,
            "default": 1,
            "yes": 1,
            "1": 1
        })

    y_test = pd.to_numeric(
        y_test,
        errors="raise"
    ).astype(int)

    # Validate target classes
    if not set(y_test.unique()).issubset({0, 1}):

        raise ValueError(
            "Test target must contain binary labels 0 and 1."
        )

    # Generate model predictions
    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)

    # Find probability column for class 1
    positive_class_index = list(
        model.classes_
    ).index(1)

    default_probabilities = y_probability[
        :, positive_class_index
    ]

    # =====================================
    # PERFORMANCE METRICS
    # =====================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # ROC-AUC requires both classes
    if len(set(y_test)) == 2:

        roc_auc = roc_auc_score(
            y_test,
            default_probabilities
        )

        fpr, tpr, thresholds = roc_curve(
            y_test,
            default_probabilities
        )

        roc_curve_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }

    else:

        roc_auc = None
        roc_curve_data = None

    # =====================================
    # CONFUSION MATRIX
    # =====================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),

        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },

        "roc_curve": roc_curve_data,

        "test_samples": int(len(y_test))

    }


# ==========================================
# 6. INITIALIZE DATABASE
# ==========================================

initialize_database()


# ==========================================
# 7. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 8. LOAN DEFAULT PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        input_data = {}

        # Collect numerical features
        for feature in NUMERICAL_FEATURES:

            value = request.form.get(
                feature, ""
            ).strip()

            if value == "":
                input_data[feature] = float("nan")

            else:
                input_data[feature] = float(value)

        # Collect categorical features
        for feature in CATEGORICAL_FEATURES:

            input_data[feature] = request.form.get(
                feature, ""
            ).strip()

        # Collect application date
        application_date = request.form.get(
            "application_date", ""
        ).strip()

        if not application_date:

            raise ValueError(
                "Please provide an application date."
            )

        input_data["application_date"] = application_date

        # Create input DataFrame
        input_df = pd.DataFrame([input_data])

        # Generate prediction
        result = prediction_pipeline.predict(input_df)

        # Prepare database input
        customer_inputs = {
            key: (
                None if pd.isna(value) else value
            )
            for key, value in input_data.items()
        }

        # Save prediction
        save_prediction(
            application_date,
            customer_inputs,
            result
        )

        return render_template(
            "result.html",
            prediction=result
        )

    except Exception as error:

        app.logger.exception(
            "Loan prediction failed"
        )

        return render_template(
            "index.html",
            error=str(error)
        )


# ==========================================
# 9. PREDICTION HISTORY
# ==========================================

@app.route("/history")
def history():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    records = connection.execute("""
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return render_template(
        "history.html",
        records=records
    )


# ==========================================
# 10. LOAN PORTFOLIO ANALYTICS DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    # Total applications
    total_applications = connection.execute("""
        SELECT COUNT(*) AS total
        FROM prediction_history
    """).fetchone()["total"]

    # Predicted defaults
    default_count = connection.execute("""
        SELECT COUNT(*) AS total
        FROM prediction_history
        WHERE prediction = 'Default'
    """).fetchone()["total"]

    # Predicted non-defaults
    non_default_count = connection.execute("""
        SELECT COUNT(*) AS total
        FROM prediction_history
        WHERE prediction = 'Non-Default'
    """).fetchone()["total"]

    # Average default probability
    average_probability = connection.execute("""
        SELECT AVG(default_probability) AS average
        FROM prediction_history
    """).fetchone()["average"]

    # Prediction distribution
    distribution_rows = connection.execute("""
        SELECT
            prediction,
            COUNT(*) AS total
        FROM prediction_history
        GROUP BY prediction
    """).fetchall()

    distribution = [
        dict(row) for row in distribution_rows
    ]

    # Recent applications
    recent_records = connection.execute("""
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    connection.close()

    # Evaluate trained model on held-out test data
    try:

        model_metrics = evaluate_model()

    except Exception:

        app.logger.exception(
            "Model evaluation failed"
        )

        model_metrics = None

    return render_template(
        "dashboard.html",

        total_applications=total_applications,

        default_count=default_count,

        non_default_count=non_default_count,

        average_probability=average_probability or 0,

        distribution=distribution,

        recent_records=recent_records,

        model_metrics=model_metrics
    )


# ==========================================
# 11. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
