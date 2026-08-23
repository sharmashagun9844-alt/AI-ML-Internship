# AI/ML Internship — Task 1

**AI/ML Development Environment Setup, Python Foundations & Data Exploration**

## Overview

This repository contains the completed deliverables for Task 1 of the AI/ML
internship: a configured development environment, Python fundamentals
practice, a cleaned dataset, and a full exploratory data analysis (EDA),
covering both the **Student Performance Data Explorer** mini project and the
**Iris dataset** advanced challenge.

## Project Structure

```
AI-ML-Internship/
│
├── data/                # Raw datasets
│   ├── student_data.csv
│   └── iris.csv
├── notebooks/           # Jupyter notebook with full analysis
│   └── AI_ML_Task1_EDA.ipynb
├── src/                 # Reusable Python scripts
│   ├── generate_student_data.py
│   ├── eda_utils.py
│   └── build_notebook.py
├── reports/             # Written findings
│   └── EDA_Report.pdf
├── images/               # Saved chart exports (PNG)
├── models/               # Reserved for Task 2 (model artifacts)
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment Setup

```bash
# 1. Check Python
python --version        # Python 3.12+

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

## Running the Analysis

```bash
# Regenerate the synthetic student dataset (optional — already included)
python src/generate_student_data.py

# Open and run notebooks/AI_ML_Task1_EDA.ipynb top to bottom
jupyter notebook notebooks/AI_ML_Task1_EDA.ipynb
```

## What's Inside the Notebook

| Section | Content |
|---|---|
| Part C | Python fundamentals demo (variables, loops, functions, data structures, file I/O) |
| Part D | Loading, inspecting, and cleaning the student dataset (missing values, duplicates, renaming) |
| Part E | Descriptive statistics, correlation heatmap, histograms, scatter plots, box plots |
| Part F | Mini Project — Student Performance Data Explorer, with a written insights summary |
| Part G | Advanced Challenge — full EDA on the Iris dataset, including a pairplot and species-level breakdown |

## Key Findings (Summary)

**Student Performance dataset**
- Study hours and attendance both correlate positively with exam scores.
- Reading and writing scores move together most strongly of any score pair.
- No significant outliers remained after cleaning; score spreads are consistent across subjects.

**Iris dataset**
- Petal length and petal width are the strongest, near-perfectly correlated predictors of species.
- *Setosa* is linearly separable from the other two species using petal measurements alone.
- *Versicolor* and *virginica* overlap slightly, so multiple features are needed to separate them cleanly.

Full details, charts, and code are in `notebooks/AI_ML_Task1_EDA.ipynb` and `reports/EDA_Report.pdf`.

## Datasets

- **student_data.csv** — a synthetic 310-row dataset (Student ID, Gender, Study Hours, Attendance %, Math/Reading/Writing Score) generated with `src/generate_student_data.py`, intentionally seeded with missing values and duplicate rows for the cleaning exercises.
- **iris.csv** — the classic 150-row Iris dataset (sepal/petal length & width, species), loaded via `scikit-learn`.

## Connection to Task 2

Task 2 builds directly on this cleaned dataset to introduce preprocessing,
feature engineering, dataset splitting, baseline model training, and
evaluation — using the same data pipeline established here.

## Author's Checklist

- [x] Environment configured (Python, VS Code, Git, Jupyter)
- [x] Virtual environment + requirements.txt
- [x] Project folder structure
- [x] Dataset loaded, cleaned, and documented
- [x] EDA notebook with statistics + visualizations
- [x] PDF report summarizing findings
- [x] README
