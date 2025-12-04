#%% Project Title
"""
Data Mining Project: Examining the Relationship Between Demographics, Admission Types, and Clinical Outcomes
Niagara County Inpatients (SPARCS 2024)
Authors: Simintaj Salehpour, Nazish Atta, Alex Biuckians
"""
#%% 
# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#%%
# ==============================
# 2. Load Dataset
# ==============================
#file_path = "Hospital_Inpatient_Discharges_2024"
df = pd.read_csv("Hospital_Inpatient_Discharges_2024.csv")

# Quick look
print(df.head())
print(df.info())
print(df.describe())





#%%
# ==============================
# 3. Data Cleaning / Preprocessing
# ==============================

# Convert numeric columns safely
numeric_cols = ['Length of Stay', 'Total Charges', 'Total Costs']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key features
key_features = ['Age Group', 'Length of Stay', 'Payment Typology 1','Total Charges', 'Total Costs']
print(f"Rows before dropna: {len(df)}")
df = df.dropna(subset=key_features)
print(f"Rows after dropna: {len(df)}")

# Encode categorical variables
categorical_features = ['Age Group', 'Gender', 'Race', 'Ethnicity', 'Type of Admission', 'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Optional: Scale numeric features
numeric_features = ['Length of Stay', 'Total Charges', 'Total Costs']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Quick check after cleaning
print(df.info())
print(df.head())





#%% 
# ==============================
# 4. Exploratory Data Analysis (EDA)
# ==============================

# Summary statistics for numeric columns
numeric_cols = ['Length of Stay', 'Total Charges', 'Total Costs']
print(df[numeric_cols].describe())

# Categorical distributions (before encoding)
categorical_cols = ['Age Group', 'Gender', 'Race', 'Ethnicity', 'Type of Admission', 
                    'Payment Typology 1','Payment Typology 2','Payment Typology 3']

for col in categorical_cols:
    if col in df.columns:
        plt.figure(figsize=(8,4))
        sns.countplot(x=col, data=df)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()

# Numeric distributions
for col in numeric_cols:
    if col in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

# Correlation heatmap for numeric features
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()





#%% 
# ==============================
# 6. SMART Question 2: For all Hospital Inpatient Discharges in Niagara County, NY in 2024 with a Primary Diagnosis Code starting with Heart Failure, how does the median Total Charges vary across all defined Age Groups in the Niagara County subset of the data? Can a regression model predict a patient’s total hospital charges for heart failure with a root mean squared error (RMSE) below $5,000, using age group, length of stay, and payer type as predictors?
#================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import altair as alt
import math

# ------------------- Load data -------------------
df = pd.read_csv("Hospital_Inpatient_Discharges_2024.csv")   # <-- CHANGE THIS

hf = df[
    (df["Hospital County"] == "Niagara") &
    (df["Discharge Year"] == 2024) &
    (df["CCSR Diagnosis Code"] == "CIR019")
]

print("Number of Heart Failure cases:", len(hf))

medians = (
    hf.groupby("Age Group")["Total Charges"]
      .agg(["count", "median", "mean", "std"])
      .reset_index()
      .sort_values("Age Group")
)

print("\n==== MEDIAN TOTAL CHARGES BY AGE GROUP (HEART FAILURE – NIAGARA COUNTY 2024) ====\n")
print(medians)

# Using: Age Group (categorical), Length of Stay (numeric), Payment Typology 1 (categorical)
hf_model = hf.dropna(subset=["Total Charges", "Length of Stay", "Payment Typology 1", "Age Group"])

X = hf_model[["Age Group", "Length of Stay", "Payment Typology 1"]]
y = hf_model["Total Charges"]

categorical_cols = ["Age Group", "Payment Typology 1"]
numeric_cols = ["Length of Stay"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

pipe_lm = Pipeline([
    ("pre", preprocessor),
    ("lm", LinearRegression())
])

cv = KFold(n_splits=10, shuffle=True, random_state=123)

lm_neg_mse = cross_val_score(
    pipe_lm, X_train, y_train, scoring="neg_mean_squared_error", cv=cv
)
lm_rmse_cv = np.sqrt(-lm_neg_mse).mean()

print("\n==== LINEAR REGRESSION PERFORMANCE ====")
print("Cross-validated RMSE:", round(lm_rmse_cv, 2))

pipe_lm.fit(X_train, y_train)
lm_preds_test = pipe_lm.predict(X_test)
lm_rmse_test = math.sqrt(mean_squared_error(y_test, lm_preds_test))
print("Test RMSE:", round(lm_rmse_test, 2))

pipe_rf = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=500, random_state=123, n_jobs=-1))
])

rf_neg_mse = cross_val_score(
    pipe_rf, X_train, y_train, scoring="neg_mean_squared_error", cv=cv
)
rf_rmse_cv = np.sqrt(-rf_neg_mse).mean()

print("\n==== RANDOM FOREST PERFORMANCE ====")
print("Cross-validated RMSE:", round(rf_rmse_cv, 2))

pipe_rf.fit(X_train, y_train)
rf_preds_test = pipe_rf.predict(X_test)
rf_rmse_test = math.sqrt(mean_squared_error(y_test, rf_preds_test))
print("Test RMSE:", round(rf_rmse_test, 2))

print("\n==== SUMMARY ====")
print("Median Charges by Age Group displayed above.")
print("Linear Regression Test RMSE:", round(lm_rmse_test, 2))
print("Random Forest Test RMSE:", round(rf_rmse_test, 2))
