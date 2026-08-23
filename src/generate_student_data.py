"""
generate_student_data.py
-------------------------
Creates a synthetic 'Student Performance' dataset for the AI/ML Task-1
mini project (Student Performance Data Explorer).

The dataset intentionally includes:
  - missing values (for Part D / missing-value handling exercises)
  - duplicate rows (for duplicate-removal exercises)
  - inconsistent column naming (for rename-column exercises)

Run:
    python src/generate_student_data.py
Output:
    data/student_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 300

genders = np.random.choice(["Male", "Female"], size=N)
study_hours = np.round(np.random.normal(4, 1.5, size=N).clip(0, 10), 1)
attendance = np.round(np.random.normal(85, 10, size=N).clip(40, 100), 1)

# Scores correlated with study hours + attendance + noise
math = (40 + study_hours * 6 + attendance * 0.25 + np.random.normal(0, 8, N)).clip(0, 100)
reading = (35 + study_hours * 5 + attendance * 0.3 + np.random.normal(0, 8, N)).clip(0, 100)
writing = (0.6 * reading + 0.4 * math + np.random.normal(0, 5, N)).clip(0, 100)

df = pd.DataFrame({
    "Student ID": range(1001, 1001 + N),
    "Name": [f"Student_{i}" for i in range(1, N + 1)],
    "Gender": genders,
    "Study Hours": study_hours,
    "Attendance (%)": attendance,
    "Math Score": np.round(math, 1),
    "Reading Score": np.round(reading, 1),
    "Writing Score": np.round(writing, 1),
})

# Inject missing values (~5%) into a few columns
for col in ["Math Score", "Reading Score", "Attendance (%)"]:
    idx = np.random.choice(df.index, size=int(0.05 * N), replace=False)
    df.loc[idx, col] = np.nan

# Inject duplicate rows (~3%)
dupes = df.sample(n=10, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/student_data.csv", index=False)
print(f"Saved data/student_data.csv with {len(df)} rows (incl. duplicates & missing values)")
