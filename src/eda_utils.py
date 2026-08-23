"""
eda_utils.py
------------
Reusable helper functions for loading, cleaning, and exploring datasets.
Used by both the Student Performance mini project and the Iris advanced
challenge notebook.
"""

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def dataset_overview(df: pd.DataFrame) -> None:
    """Print a quick overview: shape, dtypes, missing values, duplicates."""
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("Column dtypes:")
    print(df.dtypes, "\n")
    print("Missing values per column:")
    print(df.isnull().sum(), "\n")
    print(f"Duplicate rows: {df.duplicated().sum()}")


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the student performance dataset:
      - remove exact duplicate rows
      - impute missing numeric scores with column median
      - rename columns to snake_case for programmatic use
    """
    df = df.drop_duplicates().copy()

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.rename(columns={
        "Student ID": "student_id",
        "Name": "name",
        "Gender": "gender",
        "Study Hours": "study_hours",
        "Attendance (%)": "attendance_pct",
        "Math Score": "math_score",
        "Reading Score": "reading_score",
        "Writing Score": "writing_score",
    })
    return df.reset_index(drop=True)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df.describe().T
