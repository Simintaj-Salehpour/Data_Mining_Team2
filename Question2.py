#%% Project Title
"""
Data Mining Project: Examining the Relationship Between Demographics, Admission Types, and Clinical Outcomes
Niagara County Inpatients (SPARCS 2024)
Authors: Simintaj Salehpour, Nazish Atta, Alex Biuckians
"""




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

print("\n==== GOAL ASSESSMENT (RMSE < $5,000) ====")
print("Linear Regression achieved RMSE:", round(lm_rmse_test, 2))
print("Random Forest achieved RMSE:", round(rf_rmse_test, 2))

if lm_rmse_test < 5000 or rf_rmse_test < 5000:
    print("✓ A model met the goal (< $5,000 RMSE).")
else:
    print("✗ No model met the goal (< $5,000 RMSE). Charges are too variable to predict precisely.")

age_order = ["30-49", "50-69", "70 or Older"]
hf["Age Group"] = pd.Categorical(hf["Age Group"], categories=age_order, ordered=True)

# 2. Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Age Group",
    y="Total Charges",
    data=hf,
)

# 3. Apply logarithmic scale to the y-axis (crucial for skewed cost data)
plt.yscale("log") 

# 4. Add labels, title, and adjust appearance
plt.title("Distribution of Total Charges by Age Group (Log Scale)", fontsize=16)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Total Charges (Log Scale)", fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

# 5. Save the plot
plt.savefig("charges_by_age_boxplot.png")
# plt.show() # Use this only if you are in a local environment (do not use in the VM)
# %%
