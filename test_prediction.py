
import pandas as pd

from src.pipeline.predict_pipeline import LoanPredictionPipeline


# Load original dataset for a sample application
df = pd.read_csv(
    "data/raw/banking_loan_default_dataset.csv"
)

# Select one sample application
sample_application = df.drop(
    columns=[
        "loan_id",
        "customer_id",
        "default_flag"
    ]
).iloc[[0]]

# Create prediction pipeline
pipeline = LoanPredictionPipeline()

# Generate prediction
result = pipeline.predict(sample_application)

# Display result
print("LOAN DEFAULT PREDICTION")
print("-----------------------")

print("Prediction:", result["prediction"])

print("Result:", result["result"])

print(
    "Default Probability:",
    round(result["default_probability"] * 100, 2),
    "%"
)
