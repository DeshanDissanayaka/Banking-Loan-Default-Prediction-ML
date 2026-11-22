
import pandas as pd

from src.pipeline.shap_explainer import LoanShapExplainer


explainer = LoanShapExplainer()

# Load one processed test application
test_data = pd.read_csv(
    "data/processed/X_test.csv"
)

sample = test_data.iloc[[0]]

# Generate SHAP explanation
result = explainer.explain(sample)

print("\n========== SHAP EXPLANATION ==========")

print("\nFactors Increasing Risk:")
for item in result["risk_factors"]:
    print(item)

print("\nFactors Reducing Risk:")
for item in result["protective_factors"]:
    print(item)

print("\nBase Value:", result["base_value"])
