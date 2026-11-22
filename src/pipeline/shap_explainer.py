
import joblib
import shap
import pandas as pd

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "X_train.csv"
)


class LoanShapExplainer:

    def __init__(self):

        # Load trained model
        self.model = joblib.load(
            MODEL_DIR / "final_model.pkl"
        )

        # Load processed training data
        background_data = pd.read_csv(
            TRAIN_DATA_PATH
        )

        # Match model feature order
        feature_columns = list(
            self.model.feature_names_in_
        )

        background_data = background_data[
            feature_columns
        ]

        # Select representative background sample
        background_data = background_data.sample(
            n=min(100, len(background_data)),
            random_state=42
        )

        # Initialize SHAP Linear Explainer
        self.explainer = shap.LinearExplainer(
            self.model,
            background_data
        )

    def explain(self, processed_data):

        # Ensure correct feature order
        processed_data = processed_data[
            self.model.feature_names_in_
        ]

        # Generate SHAP values
        shap_explanation = self.explainer(
            processed_data
        )

        shap_values = shap_explanation.values[0]

        explanation_df = pd.DataFrame({
            "feature": processed_data.columns,
            "contribution": shap_values
        })

        # Factors increasing default risk
        risk_factors = (
            explanation_df[
                explanation_df["contribution"] > 0
            ]
            .sort_values(
                by="contribution",
                ascending=False
            )
            .head(5)
        )

        # Factors reducing default risk
        protective_factors = (
            explanation_df[
                explanation_df["contribution"] < 0
            ]
            .sort_values(
                by="contribution",
                ascending=True
            )
            .head(5)
        )

        return {
            "risk_factors": risk_factors.to_dict(
                orient="records"
            ),

            "protective_factors": protective_factors.to_dict(
                orient="records"
            ),

            "base_value": float(
                shap_explanation.base_values[0]
            )
        }
