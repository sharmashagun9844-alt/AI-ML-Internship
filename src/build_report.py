"""
build_report.py — generates reports/EDA_Report.pdf summarizing the
Student Performance and Iris exploratory data analyses.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], spaceAfter=12))
styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle(name="Bodyc", parent=styles["Normal"], fontSize=10.5, leading=15))

story = []

def h1(t): story.append(Paragraph(t, styles["H1c"]))
def h2(t): story.append(Paragraph(t, styles["H2c"]))
def body(t): story.append(Paragraph(t, styles["Bodyc"]))
def sp(h=10): story.append(Spacer(1, h))
def img(path, width=6.2*inch):
    story.append(Image(path, width=width, height=width*0.62))
    sp(10)

# --- Title ---
story.append(Paragraph("AI/ML Internship — Task 1", styles["Title"]))
story.append(Paragraph("Exploratory Data Analysis Report", styles["Heading2"]))
sp(6)
body("Task: AI/ML Development Environment Setup, Python Foundations &amp; Data Exploration")
sp(20)

# --- Section 1: Student Performance ---
h1("1. Student Performance Data Explorer")

h2("1.1 Dataset & Cleaning")
body("""The raw dataset contained 310 records with 5% missing values injected into
the Math Score, Reading Score, and Attendance columns, plus 10 duplicate rows.
Cleaning removed exact duplicates and imputed missing numeric values with the
column median, producing a consistent 300-record dataset ready for analysis.""")
sp(6)

h2("1.2 Correlation Analysis")
body("""Study hours and attendance both show a positive correlation with all three
exam scores. Reading and writing scores are the most strongly correlated pair,
reflecting shared verbal skills.""")
img("../images/student_correlation_heatmap.png")

h2("1.3 Score Distributions")
body("Math, reading, and writing scores are all roughly bell-shaped, centered "
     "in the 65-80 range, with no severe skew.")
img("../images/student_score_histograms.png")

h2("1.4 Study Hours vs Math Score")
body("A clear upward trend is visible between study hours and math score, "
     "consistent across both genders.")
img("../images/student_studyhours_vs_math.png")

h2("1.5 Outlier Check (Box Plots)")
body("Interquartile ranges are similar across the three subjects, and no "
     "extreme outliers remain after cleaning.")
img("../images/student_boxplots.png")

h2("1.6 Key Insights")
for line in [
    "Study hours and attendance are both positively associated with scores across all three subjects.",
    "Reading and writing scores move together most closely, suggesting shared underlying skills.",
    "No significant outliers remain post-cleaning; score spreads are consistent across subjects.",
    "Average scores differ only slightly by gender, indicating study habits matter more than gender in this dataset.",
]:
    body("• " + line)
sp(10)

story.append(PageBreak())

# --- Section 2: Iris ---
h1("2. Advanced Challenge — Iris Dataset EDA")

h2("2.1 Dataset Overview")
body("""The Iris dataset contains 150 records across three balanced species classes
(setosa, versicolor, virginica), each with four numeric features: sepal length,
sepal width, petal length, and petal width. The dataset has no missing values.""")
sp(6)

h2("2.2 Correlation Analysis")
body("Petal length and petal width are almost perfectly correlated, and both "
     "correlate strongly with sepal length. Sepal width is the outlier, showing "
     "a weak/negative relationship with the other features.")
img("../images/iris_correlation_heatmap.png")

h2("2.3 Feature Distributions")
img("../images/iris_histograms.png")

h2("2.4 Petal Length vs Petal Width by Species")
body("Setosa forms a tightly separated cluster, distinct from versicolor and "
     "virginica, which show partial overlap.")
img("../images/iris_scatter_petal.png")

h2("2.5 Box Plots by Species")
img("../images/iris_boxplots_by_species.png", width=6.0*inch)

h2("2.6 Pairwise Relationships")
img("../images/iris_pairplot.png", width=5.6*inch)

h2("2.7 Key Insights")
for line in [
    "Petal length and petal width are the strongest predictors of species — far more informative than sepal measurements.",
    "Setosa is linearly separable from the other two species using petal measurements alone.",
    "Versicolor and virginica overlap partially, so a classifier will need more than one feature to separate them cleanly.",
    "Sepal width carries the least discriminative information among the four features.",
]:
    body("• " + line)

sp(14)
h1("3. Conclusion")
body("""Both datasets are now cleaned, documented, and visualized, satisfying the
Task 1 deliverables. The Student Performance dataset demonstrates a full
real-world cleaning workflow (missing values + duplicates), while the Iris
dataset provides a canonical, well-behaved dataset for the advanced EDA
challenge. Both are ready to serve as inputs for baseline model training in
Task 2.""")

doc = SimpleDocTemplate("../reports/EDA_Report.pdf", pagesize=letter,
                         topMargin=0.7*inch, bottomMargin=0.7*inch,
                         leftMargin=0.75*inch, rightMargin=0.75*inch)
doc.build(story)
print("Report written to reports/EDA_Report.pdf")
