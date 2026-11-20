
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = OUTPUT_DIR / "banking_loan_default_dataset.csv"

DATA_DICTIONARY_FILE = OUTPUT_DIR / "data_dictionary.csv"

NUMBER_OF_RECORDS = 100_000

RANDOM_STATE = 42


# ============================================================
# 2. DATASET GENERATOR
# ============================================================

def generate_banking_dataset(
    number_of_records=NUMBER_OF_RECORDS,
    random_state=RANDOM_STATE
):

    rng = np.random.default_rng(random_state)

    n = number_of_records

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    age = rng.integers(21, 71, size=n)

    gender = rng.choice(
        ["Male", "Female"],
        size=n,
        p=[0.52, 0.48]
    )

    marital_status = rng.choice(
        ["Single", "Married", "Divorced", "Widowed"],
        size=n,
        p=[0.32, 0.58, 0.07, 0.03]
    )

    education_level = rng.choice(
        ["Secondary", "Diploma", "Bachelor", "Master", "PhD"],
        size=n,
        p=[0.25, 0.20, 0.35, 0.17, 0.03]
    )

    # --------------------------------------------------------
    # EMPLOYMENT INFORMATION
    # --------------------------------------------------------

    employment_type = rng.choice(
        [
            "Salaried",
            "Self-employed",
            "Business Owner",
            "Contract",
            "Unemployed"
        ],
        size=n,
        p=[0.55, 0.18, 0.13, 0.10, 0.04]
    )

    employment_years = np.minimum(
        rng.gamma(shape=2.4, scale=3.2, size=n),
        age - 18
    ).round(1)

    # --------------------------------------------------------
    # FINANCIAL INFORMATION
    # All monetary values are in LKR.
    # --------------------------------------------------------

    annual_income = rng.lognormal(
        mean=np.log(2_400_000),
        sigma=0.65,
        size=n
    )

    annual_income = np.clip(
        annual_income,
        360_000,
        30_000_000
    ).round(2)

    monthly_income = (annual_income / 12).round(2)

    existing_loans = rng.poisson(
        lam=1.2,
        size=n
    )

    existing_loans = np.clip(existing_loans, 0, 8)

    existing_debt_ratio = rng.uniform(
        0.05,
        0.45,
        size=n
    )

    monthly_existing_debt = (
        monthly_income * existing_debt_ratio
    )

    outstanding_debt = (
        monthly_existing_debt
        * 12
        * rng.uniform(1, 4, size=n)
    ).round(2)

    # --------------------------------------------------------
    # CREDIT HISTORY
    # --------------------------------------------------------

    credit_score = np.clip(
        rng.normal(680, 70, size=n),
        300,
        850
    ).round().astype(int)

    credit_history_years = np.minimum(
        rng.gamma(shape=2.5, scale=3.0, size=n),
        age - 18
    ).round(1)

    previous_defaults = rng.binomial(
        1,
        np.clip(
            0.04 + (700 - credit_score) / 1000,
            0.02,
            0.35
        ),
        size=n
    )

    delinquencies = rng.poisson(
        lam=np.clip(
            0.15 + (680 - credit_score) / 180,
            0.05,
            1.5
        ),
        size=n
    )

    delinquencies = np.clip(delinquencies, 0, 8)

    credit_utilization = np.clip(
        rng.beta(2.2, 3.5, size=n),
        0,
        1
    ).round(3)

    # --------------------------------------------------------
    # LOAN INFORMATION
    # --------------------------------------------------------

    loan_purpose = rng.choice(
        [
            "Personal",
            "Housing",
            "Vehicle",
            "Education",
            "Business",
            "Medical",
            "Debt Consolidation"
        ],
        size=n,
        p=[0.25, 0.15, 0.18, 0.10, 0.12, 0.08, 0.12]
    )

    loan_amount = (
        annual_income
        * rng.uniform(0.15, 2.2, size=n)
    )

    loan_amount = np.clip(
        loan_amount,
        100_000,
        15_000_000
    ).round(2)

    loan_term_months = rng.choice(
        [12, 24, 36, 48, 60, 84, 120],
        size=n,
        p=[0.08, 0.12, 0.18, 0.15, 0.25, 0.12, 0.10]
    )

    interest_rate = (
        10
        + (700 - credit_score) * 0.025
        + previous_defaults * 2.0
        + credit_utilization * 3
        + rng.normal(0, 1.5, size=n)
    )

    interest_rate = np.clip(
        interest_rate,
        6,
        28
    ).round(2)

    # --------------------------------------------------------
    # MONTHLY EMI CALCULATION
    # --------------------------------------------------------

    monthly_interest_rate = interest_rate / 1200

    emi_factor = (
        (1 + monthly_interest_rate) ** loan_term_months
    )

    monthly_emi = (
        loan_amount
        * monthly_interest_rate
        * emi_factor
        / (emi_factor - 1)
    )

    monthly_emi = monthly_emi.round(2)

    debt_to_income_ratio = (
        (monthly_existing_debt + monthly_emi)
        / monthly_income
    ).round(4)

    # --------------------------------------------------------
    # CUSTOMER SECURITY INFORMATION
    # --------------------------------------------------------

    home_ownership = rng.choice(
        ["Own", "Mortgage", "Rent", "Other"],
        size=n,
        p=[0.30, 0.32, 0.30, 0.08]
    )

    has_cosigner = rng.choice(
        ["Yes", "No"],
        size=n,
        p=[0.25, 0.75]
    )

    collateral_type = rng.choice(
        ["None", "Property", "Vehicle", "Fixed Deposit"],
        size=n,
        p=[0.40, 0.30, 0.20, 0.10]
    )

    collateral_value = np.zeros(n)

    secured_mask = collateral_type != "None"

    collateral_value[secured_mask] = (
        loan_amount[secured_mask]
        * rng.uniform(0.5, 2.0, size=secured_mask.sum())
    )

    collateral_value = collateral_value.round(2)

    # --------------------------------------------------------
    # APPLICATION DATE
    # --------------------------------------------------------

    start_date = np.datetime64("2019-01-01")
    end_date = np.datetime64("2026-01-01")

    number_of_days = int(
        (end_date - start_date) / np.timedelta64(1, "D")
    )

    application_dates = (
        start_date
        + rng.integers(0, number_of_days, size=n)
        .astype("timedelta64[D]")
    )

    application_dates = pd.to_datetime(
        application_dates
    ).strftime("%Y-%m-%d")

    # --------------------------------------------------------
    # DEFAULT TARGET GENERATION
    # --------------------------------------------------------

    unemployed_risk = (employment_type == "Unemployed").astype(int)

    contract_risk = (employment_type == "Contract").astype(int)

    log_odds = (
        -2.1
        + ((650 - credit_score) / 90) * 0.65
        + (debt_to_income_ratio - 0.35) * 3.0
        + previous_defaults * 1.05
        + delinquencies * 0.28
        + (interest_rate - 14) * 0.055
        + ((loan_amount / annual_income) - 0.8) * 0.35
        + unemployed_risk * 0.55
        + contract_risk * 0.15
        + credit_utilization * 0.65
    )

    default_probability = (
        1 / (1 + np.exp(-log_odds))
    )

    default_flag = rng.binomial(
        1,
        default_probability,
        size=n
    )

    # --------------------------------------------------------
    # CREATE FINAL DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame({

        "loan_id": [
            f"LN{i:07d}" for i in range(1, n + 1)
        ],

        "customer_id": [
            f"CUST{i:07d}" for i in range(1, n + 1)
        ],

        "application_date": application_dates,

        "age": age,

        "gender": gender,

        "marital_status": marital_status,

        "education_level": education_level,

        "employment_type": employment_type,

        "employment_years": employment_years,

        "annual_income_lkr": annual_income,

        "monthly_income_lkr": monthly_income,

        "home_ownership": home_ownership,

        "existing_loans": existing_loans,

        "outstanding_debt_lkr": outstanding_debt,

        "credit_score": credit_score,

        "credit_history_years": credit_history_years,

        "previous_defaults": previous_defaults,

        "delinquencies_last_2yrs": delinquencies,

        "credit_utilization": credit_utilization,

        "loan_purpose": loan_purpose,

        "loan_amount_lkr": loan_amount,

        "loan_term_months": loan_term_months,

        "interest_rate_pct": interest_rate,

        "monthly_emi_lkr": monthly_emi,

        "debt_to_income_ratio": debt_to_income_ratio,

        "has_cosigner": has_cosigner,

        "collateral_type": collateral_type,

        "collateral_value_lkr": collateral_value,

        "default_flag": default_flag

    })

    # --------------------------------------------------------
    # INTRODUCE CONTROLLED MISSING VALUES
    # --------------------------------------------------------

    missing_columns = [
        "employment_years",
        "credit_score",
        "credit_history_years",
        "home_ownership",
        "loan_purpose",
        "annual_income_lkr"
    ]

    missing_rates = [0.025, 0.015, 0.020, 0.020, 0.015, 0.015]

    for column, rate in zip(missing_columns, missing_rates):

        missing_mask = rng.random(n) < rate

        df.loc[missing_mask, column] = np.nan

    return df


# ============================================================
# 3. DATA DICTIONARY
# ============================================================

def create_data_dictionary():

    columns = [
        ("loan_id", "Unique loan identifier", "Identifier"),
        ("customer_id", "Unique customer identifier", "Identifier"),
        ("application_date", "Loan application date", "Date"),
        ("age", "Borrower age in years", "Numerical"),
        ("gender", "Borrower gender", "Categorical"),
        ("marital_status", "Borrower marital status", "Categorical"),
        ("education_level", "Highest education level", "Categorical"),
        ("employment_type", "Employment category", "Categorical"),
        ("employment_years", "Years of employment", "Numerical"),
        ("annual_income_lkr", "Annual income in LKR", "Numerical"),
        ("monthly_income_lkr", "Monthly income in LKR", "Numerical"),
        ("home_ownership", "Housing ownership status", "Categorical"),
        ("existing_loans", "Number of existing loans", "Numerical"),
        ("outstanding_debt_lkr", "Outstanding debt in LKR", "Numerical"),
        ("credit_score", "Borrower credit score", "Numerical"),
        ("credit_history_years", "Years of credit history", "Numerical"),
        ("previous_defaults", "Previous default indicator", "Binary"),
        ("delinquencies_last_2yrs", "Delinquencies in previous two years", "Numerical"),
        ("credit_utilization", "Credit utilization ratio", "Numerical"),
        ("loan_purpose", "Purpose of the loan", "Categorical"),
        ("loan_amount_lkr", "Requested loan amount in LKR", "Numerical"),
        ("loan_term_months", "Loan repayment term in months", "Numerical"),
        ("interest_rate_pct", "Annual interest rate percentage", "Numerical"),
        ("monthly_emi_lkr", "Monthly installment in LKR", "Numerical"),
        ("debt_to_income_ratio", "Monthly debt obligations divided by income", "Numerical"),
        ("has_cosigner", "Whether loan has a cosigner", "Categorical"),
        ("collateral_type", "Type of loan collateral", "Categorical"),
        ("collateral_value_lkr", "Estimated collateral value in LKR", "Numerical"),
        ("default_flag", "1 = Default, 0 = Non-default", "Target")
    ]

    return pd.DataFrame(
        columns,
        columns=["column_name", "description", "data_type"]
    )


# ============================================================
# 4. SAVE DATASET AND DATA DICTIONARY
# ============================================================

def main():

    print("=" * 65)
    print("BANKING LOAN DEFAULT DATASET GENERATOR")
    print("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nGenerating banking loan records...")

    df = generate_banking_dataset()

    print("Saving dataset...")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Creating data dictionary...")

    dictionary = create_data_dictionary()

    dictionary.to_csv(
        DATA_DICTIONARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # DATASET VALIDATION
    # --------------------------------------------------------

    assert len(df) == NUMBER_OF_RECORDS

    assert df["loan_id"].nunique() == NUMBER_OF_RECORDS

    assert df["customer_id"].nunique() == NUMBER_OF_RECORDS

    assert df["default_flag"].isin([0, 1]).all()

    # --------------------------------------------------------
    # DISPLAY GENERATION SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("DATASET GENERATED SUCCESSFULLY!")

    print("=" * 65)

    print(f"\nNumber of records : {len(df):,}")

    print(f"Number of columns : {len(df.columns)}")

    print(
        f"Default rate      : {df['default_flag'].mean() * 100:.2f}%"
    )

    print(
        f"Non-default       : {(df['default_flag'] == 0).sum():,}"
    )

    print(
        f"Default           : {(df['default_flag'] == 1).sum():,}"
    )

    print(
        f"\nDataset size      : {OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB"
    )

    print(f"\nDataset saved to:\n{OUTPUT_FILE}")

    print(f"\nDictionary saved to:\n{DATA_DICTIONARY_FILE}")

    print("\nMissing values by column:")

    print(
        df.isnull().sum()[df.isnull().sum() > 0]
    )

    print("\nFirst five records:")

    print(df.head())

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
