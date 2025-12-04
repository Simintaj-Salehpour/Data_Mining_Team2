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
import re
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score



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
# Convert numeric columns safely (does NOT modify structure)
numeric_cols = ['Length of Stay', 'Total Charges', 'Total Costs']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key features (keeps structure the same)
key_features = ['Age Group', 'Length of Stay', 'Payment Typology 1',
                'Total Charges', 'Total Costs']

print(f"Rows before dropna: {len(df)}")
df = df.dropna(subset=key_features)
print(f"Rows after dropna: {len(df)}")


# Quick check after cleaning
print(df.info())


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
# 5. SMART Question 1: What combination of patient demographics, admission type, and comorbidities predicts the lowest hospital charges for Medicaid vs. Private Insurance patients in 2024?
# ==============================

print(" SMART Question 1: What combination of patient demographics, admission type, and comorbidities predicts the lowest hospital charges for Medicaid vs. Private Insurance patients in 2024?")

# CLEANING
df["Total Charges"] = (
    df["Total Charges"]
    .astype(str)
    .str.replace(r"[^0-9.]", "", regex=True)
    .astype(float)
)

df["Length of Stay"] = pd.to_numeric(df["Length of Stay"], errors="coerce")

clean_data = df[
    df["Total Charges"].notna() &
    np.isfinite(df["Total Charges"]) &
    (df["Total Charges"] > 0)
]

# INSURANCE LABELING
payment_cols = [c for c in df.columns if c.startswith("Payment Typology")]
print("Payment columns:", payment_cols)
medicaid_pattern = r"(Medicaid|federal/state/local/va)"
private_pattern  = r"(Blue Cross/Blue Shield|Private Health Insurance|Miscellaneous/Other)"
dpayment_cols = [c for c in df.columns if c.startswith("Payment Typology")]

def assign_insurance(row):
    combined = " ".join([str(row[col]).lower() for col in payment_cols])
    if ("medicaid" in combined) or ("federal/state/local/va" in combined):
        return "Medicaid"

    if ("private health insurance" in combined) or ("blue cross" in combined) or ("miscellaneous/other" in combined):
        return "Private"
    return "Other"

clean_data['Insurance'] = df.apply(assign_insurance, axis=1)
print("Insurance counts:\n", clean_data['Insurance'].value_counts())
print("After Insurance filtering:", len(df))

# KEEP RELEVANT COLUMNS
columns_to_keep = [
  'Age Group', 'Gender','Race', 'Length of Stay','Type of Admission',
  'Emergency Department Indicator','APR Severity of Illness Description',
  'APR DRG Code','CCSR Diagnosis Description',
  'Insurance','Total Charges'
]

hospital_final = clean_data[columns_to_keep].copy()

# LUMP RARE LEVELS (fct_lump_min)
def lump_min(series, min_count=20):
    counts = series.value_counts()
    rare = counts[counts < min_count].index
    return series.replace(rare, "Other").fillna("Other")

factors_to_fix = [
    "Age Group","Gender","Race","Type of Admission",
    "Emergency Department Indicator",
    "APR Severity of Illness Description",
    "CCSR Diagnosis Description"
]

for f in factors_to_fix:
    hospital_final[f] = lump_min(hospital_final[f].astype(str))

# REMOVE NA PREDICTORS
predictors = [
    'Age Group', 'Gender','Race', 'Length of Stay','Type of Admission',
  'Emergency Department Indicator','APR Severity of Illness Description',
  'APR DRG Code','CCSR Diagnosis Description'
]

hospital_final = hospital_final.dropna(subset=predictors)

# SPLIT BY INSURANCE
Medicaid_df = hospital_final[hospital_final["Insurance"] == "Medicaid"]
Private_df  = hospital_final[hospital_final["Insurance"] == "Private"]

Medicaid_df = Medicaid_df.copy()
Private_df  = Private_df.copy()

# Define categorical columns
cat_cols = ["Age Group", "Gender", "Race", "Type of Admission",
            "APR Severity of Illness Description", "CCSR Diagnosis Description"]

# Clean columns
for df in [Medicaid_df, Private_df]:
    # Map Y/N to 1/0
    df["Emergency Department Indicator"] = df["Emergency Department Indicator"].map({"Y": 1, "N": 0})
    # Ensure Total Charges numeric
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    # Strip whitespace and convert to string for categorical columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

# Linear Regression
def run_linear_regression(df, group_name):
    features = ["Length of Stay", "Emergency Department Indicator"] + cat_cols
    X = pd.get_dummies(df[features], drop_first=True).astype(float)
    y = df["Total Charges"].astype(float)

    # Drop rows with NaN
    valid = (~X.isnull().any(axis=1)) & (~y.isnull()) & np.isfinite(y)
    X = X.loc[valid]
    y = y.loc[valid]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    print(f"\n===== LINEAR REGRESSION ({group_name}) =====")
    print(model.summary())
    return model


# Logistic Regression (GLM)
def run_logistic_regression(df, group_name):
    median_charge = df["Total Charges"].median()
    df["LowCharge"] = (df["Total Charges"] <= median_charge).astype(int)

    features = ["Length of Stay", "Emergency Department Indicator"] + cat_cols
    X = pd.get_dummies(df[features], drop_first=True).astype(float)
    y = df["LowCharge"]

    valid = (~X.isnull().any(axis=1)) & (~y.isnull()) & np.isfinite(y)
    X = X.loc[valid]
    y = y.loc[valid]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== LOGISTIC REGRESSION ({group_name}) =====")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    # GLM with p-values
    X_glm = sm.add_constant(X)
    glm_model = sm.GLM(y, X_glm, family=sm.families.Binomial()).fit()
    print("\n--- GLM with p-values ---")
    print(glm_model.summary())
    return glm_model

# Random Forest
def run_random_forest(df, group_name):
    features = ["Length of Stay", "Emergency Department Indicator"] + cat_cols
    X = df[features]
    y = df["Total Charges"].astype(float)

    valid = X.notnull().all(axis=1) & y.notnull() & np.isfinite(y)
    X = X.loc[valid]
    y = y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Get dummies on train
    X_train_dummies = pd.get_dummies(X_train, drop_first=True)
    # Apply same columns to test, fill missing with 0
    X_test_dummies = pd.get_dummies(X_test, drop_first=True)
    X_test_dummies = X_test_dummies.reindex(columns=X_train_dummies.columns, fill_value=0)

    # Fit Random Forest
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_train_dummies, y_train)
    y_pred = rf.predict(X_test_dummies)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n===== RANDOM FOREST ({group_name}) =====")
    print("Mean Squared Error:", mse)
    print("R²:", r2)

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X_train_dummies.columns)
    # Aggregate importance by original feature (e.g., sum dummy contributions)
    agg_importances = importances.groupby(lambda x: x.split("_")[0] if "_" in x else x).sum()
    agg_importances = agg_importances.sort_values(ascending=False)

    print("Top 10 Features by Importance:\n", agg_importances.head(10))

    # Plot top 10 features
    plt.figure(figsize=(10,6))
    sns.barplot(x=agg_importances.head(15), y=agg_importances.head(15).index)
    plt.title(f"Top Features - Random Forest ({group_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return agg_importances


#Run linear model, logistic regression and random forest
for df, name in zip([Medicaid_df, Private_df], ["Medicaid", "Private"]):
    run_linear_regression(df, name)
    run_logistic_regression(df, name)
    run_random_forest(df, name)


print("Results:\n")
print("Linear Regression shows good model fit for Medicaid and moderate for Private.")
print("Logistic Regression is more accurate for Medicaid low charges prediction.")
print("Random Forest identifies Length of Stay and Diagnosis/Severity as dominant predictors.")
print("Overall, Medicaid hospital charges are better predicted than Private charges.")


#%%
# ==============================
# 6. SMART Question 2: For all Hospital Inpatient Discharges in Niagara County, NY in 2024 with a Primary Diagnosis Code starting with Heart Failure, how does the median Total Charges vary across all defined Age Groups in the Niagara County subset of the data? Can a regression model predict a patient’s total hospital charges for heart failure with a root mean squared error (RMSE) below $5,000, using age group, length of stay, and payer type as predictors?
# ==============================
















#%% 
# ==============================
# 7. SMART Question 3: How does patient age affect hospitalization factors like Length of Stay and Total Charges across various age groups in 2024?
# ==============================



















# ==============END================

