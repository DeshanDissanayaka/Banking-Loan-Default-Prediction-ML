# Banking Loan Default Prediction – End-to-End Machine Learning Project

## Project Overview

This project uses machine learning to predict whether a bank loan applicant is likely to default. It covers the main stages of a machine learning workflow, from data preparation and model training to making predictions through a Flask web application.

The project uses a synthetic banking loan dataset and a Logistic Regression model. The web application accepts applicant and loan information, returns a prediction with the estimated default probability, and provides SHAP-based explanations for the factors influencing the result.

## Key Features

- Exploratory data analysis and data preprocessing
- Handling missing numerical and categorical values
- Encoding categorical variables and scaling numerical features
- Logistic Regression model training and evaluation
- Loan default predictions with estimated probabilities
- SHAP-based explanations of risk-increasing and risk-reducing factors
- Flask web application for submitting loan applications
- SQLite database for storing prediction history
- Dashboard showing prediction activity and model evaluation results

## Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Chart.js |
| Machine learning | Scikit-learn |
| Model explainability | SHAP |
| Web application | Flask, HTML, CSS, JavaScript |
| Database | SQLite |
| Development | Jupyter Notebook, VS Code, Git, GitHub |

## Project Structure

```text
Banking-Loan-Default-Prediction-ML/
│
├── artifacts/
│   └── models/
│       ├── numerical_imputer.pkl
│       ├── categorical_imputer.pkl
│       ├── onehot_encoder.pkl
│       ├── standard_scaler.pkl
│       └── final_model.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebook/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   └── 03_Model_Training.ipynb
│
├── src/
│   ├── components/
│   └── pipeline/
│       ├── predict_pipeline.py
│       └── shap_explainer.py
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   └── dashboard.html
│
├── application.py
├── requirements.txt
├── setup.py
└── README.md
```

## Machine Learning Workflow

1. **Exploratory data analysis:** Reviewed the dataset, feature distributions and relationships between applicant characteristics and loan default.
2. **Data preprocessing:** Handled missing values, transformed categorical variables, scaled numerical features and prepared the training and test datasets.
3. **Model training:** Trained a Logistic Regression classifier using the prepared training data.
4. **Model evaluation:** Evaluated the model using the test dataset and classification metrics.
5. **Prediction pipeline:** Applied the saved preprocessing objects and trained model to new applicant information.
6. **Model explainability:** Used SHAP to identify the features contributing to each prediction.
7. **Web application:** Built a Flask application to submit applications, view predictions and review prediction history and model performance.

## Model Performance

The following results were obtained from the test dataset:

| Metric | Result |
|---|---:|
| Test records | 20,000 |
| Accuracy | 74.64% |
| Precision | 60.70% |
| Recall | 71.18% |
| F1-score | 65.52% |
| ROC-AUC | 0.8175 |

These metrics describe the model's performance on the prepared test dataset. They should not be interpreted as evidence of performance on real bank customers.

## Model Explainability

SHAP is used to explain individual predictions by showing which input features contributed to increasing or reducing the model's predicted default risk.

The result page displays the leading risk-increasing and risk-reducing factors for an application. These explanations describe the model's behaviour; they do not establish causal relationships.

## Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/DeshanDissanayaka/Banking-Loan-Default-Prediction-ML.git
cd Banking-Loan-Default-Prediction-ML
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask application

```bash
python application.py
```

Open the local address displayed in the terminal in your browser.

## Disclaimer

This project was developed for learning and demonstration purposes using synthetic data. It is not intended for actual lending decisions or use as a production banking credit-risk system.

## Author

**Deshan Dissanayaka**

[GitHub Repository](https://github.com/DeshanDissanayaka/Banking-Loan-Default-Prediction-ML)