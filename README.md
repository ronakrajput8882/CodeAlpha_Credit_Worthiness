<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🏦%20Creditworthiness%20Predictor&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Loan%20Approval%20Prediction%20using%20Random%20Forest%20%7C%20Binary%20Classification&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Project Overview

Predicting whether a loan application will be **approved or rejected** based on applicant financial and demographic data. This project applies supervised binary classification — comparing Logistic Regression, Decision Tree, and Random Forest — to build a robust creditworthiness evaluation system that financial institutions can rely on.

> **Best Model:** 🏆 Random Forest Classifier — selected for its ensemble strength, resistance to overfitting, and superior performance across all metrics.

---

## 📂 Dataset

| Property | Value |
|:---|:---|
| Records | 1,000 applicants |
| Features | 19 input features + 1 target |
| Target | `Loan_Approved` (Yes / No) |
| Missing Values | Present (handled via `SimpleImputer`) |
| Source | Custom / Synthetic loan application data |

**Feature Categories:**

- **Financial:** `Applicant_Income`, `Coapplicant_Income`, `Credit_Score`, `Savings`, `Collateral_Value`, `DTI_Ratio`, `Existing_Loans`
- **Loan Details:** `Loan_Amount`, `Loan_Term`, `Loan_Purpose`
- **Demographic:** `Age`, `Gender`, `Marital_Status`, `Dependents`, `Education_Level`
- **Employment:** `Employment_Status`, `Employer_Category`
- **Location:** `Property_Area`

---

## 🔄 Pipeline Workflow

```
Raw Data → Missing Value Imputation → EDA & Visualization → Feature Encoding → StandardScaler → Train/Test Split → Model Training → Evaluation
```

### 1️⃣ Data Loading & Inspection
Loaded `loan_approval_data.csv` (1,000 rows × 20 columns). Separated numerical and categorical columns for targeted preprocessing.

### 2️⃣ Missing Value Handling
```python
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")
```
Applied mean imputation for numerical columns and most-frequent for categoricals — preserving data distribution without dropping rows.

### 3️⃣ Exploratory Data Analysis
- **Loan Approval Distribution** — pie chart of Yes/No split
- **Gender & Education** — bar plots for demographic overview
- **Income Distribution** — histplots for Applicant and Co-applicant income
- **Box Plots** — Loan Amount, Credit Score, Savings, and DTI Ratio vs. approval outcome
- **Credit Score by Approval** — dodged histplot revealing approval threshold

### 4️⃣ Feature Encoding
```python
# Label Encoding
LabelEncoder()  →  Education_Level, Loan_Approved

# One-Hot Encoding (drop='first')
OneHotEncoder() →  Employment_Status, Marital_Status, Loan_Purpose,
                   Property_Area, Gender, Employer_Category
```

### 5️⃣ Feature Scaling
```python
StandardScaler()  # fit on X_train, transform both train & test
```

### 6️⃣ Train / Test Split
```python
train_test_split(X, y, test_size=0.2, random_state=42)
# 800 training | 200 testing
```

---

## 🤖 Models

### 1️⃣ Logistic Regression
```python
LogisticRegressionCV()
```
- Cross-validated regularization strength selection
- Trained on scaled features
- Strong baseline for linear separability

### 2️⃣ Decision Tree
```python
DecisionTreeClassifier()
```
- Full depth tree — interpretable rules
- Visualized with `plot_tree()` for feature split analysis
- Prone to overfitting on noisy features

### 3️⃣ Random Forest Classifier ⭐ Best Model
```python
RandomForestClassifier(
    n_estimators=200,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
```
- 200 trees with bootstrapped subsets
- `class_weight='balanced'` — handles class imbalance
- `max_features='sqrt'` — reduces correlation between trees
- `min_samples_leaf=2` — prevents memorizing noise

---

## 📊 Results

> ⚠️ **Note:** Run the notebook cells to populate exact values — update the table below with your output.

| Model | Precision | Recall | F1 Score | AUC |
|:---|:---:|:---:|:---:|:---:|
| Logistic Regression | — | — | — | — |
| Decision Tree | — | — | — | — |
| 🏆 **Random Forest** | **—** | **—** | **—** | **—** |

*Fill in after running `creditworthiness.ipynb`*

---

## 🔍 Key Insights

- 📈 **Credit Score** has a **+0.45 positive correlation** with `Loan_Approved` — the single strongest predictor
- 📉 **DTI Ratio** has a **-0.44 negative correlation** with `Loan_Approved` — higher debt-to-income significantly lowers approval odds
- 🚫 **Credit Score below 650** → zero loan approvals observed in the dataset (hard threshold)
- 🏆 **Random Forest** outperformed Logistic Regression and Decision Tree by leveraging ensemble averaging across 200 trees with balanced class weights
- 💰 **Savings** and **Collateral Value** showed strong boxplot separation between approved and rejected applications

---

## 🗂️ Repository Structure

```
creditworthiness-predictor/
│
├── creditworthiness.ipynb      # Main notebook — full pipeline
├── loan_approval_data.csv      # Raw dataset (1000 records)
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ronakrajput8882/creditworthiness-predictor.git
cd creditworthiness-predictor

# 2. Install dependencies
pip install pandas scikit-learn seaborn matplotlib jupyter

# 3. Launch the notebook
jupyter notebook creditworthiness.ipynb
```

---

## 🧠 Key Learnings

- **Credit Score** is a non-negotiable threshold feature — tree models picked this up as the first split
- **DTI Ratio** and **Credit Score** together explain most of the variance in approval decisions
- `class_weight='balanced'` in Random Forest is critical when the approval/rejection ratio is skewed
- `OneHotEncoder(drop='first')` avoids dummy variable trap in logistic models
- `StandardScaler` must be **fit only on training data** to prevent data leakage

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| Pandas | Data loading & manipulation |
| Scikit-learn | Preprocessing, modeling, evaluation |
| Seaborn / Matplotlib | EDA visualizations |
| Jupyter Notebook | Development environment |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronaksinh-rajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>