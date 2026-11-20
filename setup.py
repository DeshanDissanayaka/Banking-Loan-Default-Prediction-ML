
from setuptools import find_packages, setup

setup(
    name="banking-loan-default-prediction",
    version="0.1.0",
    author="Deshan Dissanayaka",
    description="End-to-end Banking Loan Default Prediction using Machine Learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[],
)
