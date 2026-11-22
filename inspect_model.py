
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

numerical_imputer = joblib.load(
    MODEL_DIR / "numerical_imputer.pkl"
)

categorical_imputer = joblib.load(
    MODEL_DIR / "categorical_imputer.pkl"
)

model = joblib.load(
    MODEL_DIR / "final_model.pkl"
)

print("\n========== NUMERICAL FEATURES ==========")

for column in numerical_imputer.feature_names_in_:
    print(column)

print("\n========== CATEGORICAL FEATURES ==========")

for column in categorical_imputer.feature_names_in_:
    print(column)

print("\n========== FINAL MODEL FEATURES ==========")

for column in model.feature_names_in_:
    print(column)

print("\n========== TOTAL FEATURE COUNTS ==========")

print("Numerical:", len(numerical_imputer.feature_names_in_))
print("Categorical:", len(categorical_imputer.feature_names_in_))
print("Final model:", len(model.feature_names_in_))
