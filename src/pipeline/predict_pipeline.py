
import pandas as pd
import joblib

from pathlib import Path

from src.pipeline.shap_explainer import LoanShapExplainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"


class LoanPredictionPipeline:

    def __init__(self):

        self.numerical_imputer = joblib.load(
            MODEL_DIR / "numerical_imputer.pkl"
        )

        self.categorical_imputer = joblib.load(
            MODEL_DIR / "categorical_imputer.pkl"
        )

        self.encoder = joblib.load(
            MODEL_DIR / "onehot_encoder.pkl"
        )

        self.scaler = joblib.load(
            MODEL_DIR / "standard_scaler.pkl"
        )

        self.model = joblib.load(
            MODEL_DIR / "final_model.pkl"
        )

        # Initialize SHAP explainer
        self.shap_explainer = LoanShapExplainer()

    def predict(self, input_data):

        data = input_data.copy()

        # Percentage conversion
        percentage_columns = [
            "credit_utilization",
            "debt_to_income_ratio"
        ]

        for column in percentage_columns:

            if column in data.columns:

                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
                ) / 100

        # Date feature engineering
        data["application_date"] = pd.to_datetime(
            data["application_date"]
        )

        data["application_year"] = (
            data["application_date"].dt.year
        )

        data["application_month"] = (
            data["application_date"].dt.month
        )

        data["application_day"] = (
            data["application_date"].dt.day
        )

        data["application_dayofweek"] = (
            data["application_date"].dt.dayofweek
        )

        data.drop(
            columns=["application_date"],
            inplace=True
        )

        # Identify preprocessing columns
        numerical_columns = list(
            self.numerical_imputer.feature_names_in_
        )

        categorical_columns = list(
            self.categorical_imputer.feature_names_in_
        )

        # Numerical imputation
        numerical_data = pd.DataFrame(
            self.numerical_imputer.transform(
                data[numerical_columns]
            ),
            columns=numerical_columns,
            index=data.index
        )

        # Categorical imputation
        categorical_data = pd.DataFrame(
            self.categorical_imputer.transform(
                data[categorical_columns]
            ),
            columns=categorical_columns,
            index=data.index
        )

        # One-hot encoding
        encoded_data = self.encoder.transform(
            categorical_data
        )

        encoded_columns = (
            self.encoder.get_feature_names_out(
                categorical_columns
            )
        )

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoded_columns,
            index=data.index
        )

        # Combine features
        processed_data = pd.concat(
            [numerical_data, encoded_df],
            axis=1
        )

        # Feature scaling
        processed_data[numerical_columns] = (
            self.scaler.transform(
                processed_data[numerical_columns]
            )
        )

        # Match model feature order
        processed_data = processed_data[
            self.model.feature_names_in_
        ]

        # Prediction
        prediction = self.model.predict(
            processed_data
        )

        probability = self.model.predict_proba(
            processed_data
        )[:, 1]

        # SHAP explanation
        shap_result = self.shap_explainer.explain(
            processed_data
        )

        return {

            "prediction": int(prediction[0]),

            "default_probability": float(
                probability[0]
            ),

            "result": (
                "Default"
                if prediction[0] == 1
                else "Non-Default"
            ),

            "risk_factors": shap_result[
                "risk_factors"
            ],

            "protective_factors": shap_result[
                "protective_factors"
            ],

            "shap_base_value": shap_result[
                "base_value"
            ]
        }
