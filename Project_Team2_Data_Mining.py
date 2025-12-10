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
clean_data['ChargePerDay']=clean_data['Total Charges']/clean_data['Length of Stay']

# KEEP RELEVANT COLUMNS
columns_to_keep = [
  'Age Group', 'Gender','Race', 'Length of Stay','Type of Admission',
  'Emergency Department Indicator','APR Severity of Illness Description',
  'APR DRG Code','CCSR Diagnosis Description',
  'Insurance','Total Charges','ChargePerDay'
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
    features = ["Emergency Department Indicator"] + cat_cols
    X = df[features]
    y = df["ChargePerDay"].astype(float)

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




#%% 
# ==============================
# 7. SMART Question 3: How does patient age affect hospitalization factors like Length of Stay and Total Charges across various age groups in 2024?
# ==============================

# ------------------------------------------
# Question #3 Analysis Script
# How does patient age (measured by Age Group)
# affect Length of Stay and Total Charges in 2024 (Niagara County)?
# ------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------
# 1. LOAD DATA
# ------------
raw_df = pd.read_csv("Hospital_Inpatient_Discharges_2024.csv")
df = raw_df[raw_df["Hospital County"] == "Niagara"].copy()


# --------------------------
# 2. SELECT RELEVANT COLUMNS
# --------------------------

#  the dataset has "Age Group" so  patient age (measured by Age Group)
cols = ["Age Group", "Length of Stay", "Total Charges"]
df = df[cols]

# ----------------
# 3. DATA CLEANING
# ----------------

# Convert numeric columns
df["Length of Stay"] = pd.to_numeric(df["Length of Stay"], errors="coerce")
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# Drop rows with missing values in these columns
df = df.dropna(subset=["Age Group", "Length of Stay", "Total Charges"])

# Remove invalid LOS or charges (<= 0)
df = df[df["Length of Stay"] > 0]
df = df[df["Total Charges"] > 0]

# -----------------------
# 4. EXPLORATORY ANALYSIS
# -----------------------

# Median Length of Stay by Age Group
los_stats = df.groupby("Age Group")["Length of Stay"].median()
print("Median Length of Stay by Age Group:")
print(los_stats)
print("\n")

# Median Total Charges by Age Group
charges_stats = df.groupby("Age Group")["Total Charges"].median()
print("Median Total Charges by Age Group:")
print(charges_stats)
print("\n")

# Correlation between Length of Stay and Total Charges
corr = df["Length of Stay"].corr(df["Total Charges"])
print(f"Correlation between Length of Stay and Total Charges: {corr:.2f}")
print("\n")

print(f"""The correlation coefficient of approximately {corr:.2f} 
      indicates a strong positive linear relationship between
      Length of Stay and Total Charges — meaning longer hospital 
      stays are typically associated with higher total costs."""
      )

# -------------------------------
# 5. VISUALIZATIONS
# -------------------------------

#  CORRELATION VISUALIZATION

plt.figure(figsize=(8, 6))
plt.text(5, 300000, "r = 0.78 (Strong Positive Correlation)", color="red", fontsize=12)

sns.scatterplot(data=df, x="Length of Stay", y="Total Charges", alpha=0.6)
sns.regplot(data=df, x="Length of Stay", y="Total Charges", scatter=False, color="red")


plt.title("Correlation Between Length of Stay and Total Charges (Niagara County, 2024)")
plt.xlabel("Length of Stay (days)")
plt.ylabel("Total Charges ($)")
plt.tight_layout()
plt.show()

print(
    f"""
    Interpretation:
    This scatterplot shows a clear upward trend — as Length of Stay increases,
    Total Charges also rise. The red regression line confirms a strong
    positive linear relationship, consistent with our correlation value of {corr:.2f}.
    Longer hospital stays generally lead to higher treatment costs.
    """
)


print(df["Age Group"].value_counts())

# order for Age Group 
age_order = ["0-17", "18-29", "30-49", "50-69", "70 or Older"]

# Boxplot: Length of Stay by Age Group
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Age Group", y="Length of Stay", order=age_order)
plt.title("Length of Stay by Age Group (Niagara County, 2024)")
plt.xlabel("Age Group")
plt.ylabel("Length of Stay (days)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(
    """
    Interpretation:
    The boxplot shows that younger patients (0–17) generally have the shortest hospital stays,
    while middle-aged and older adults (30–69) tend to remain hospitalized longer.
    The median length of stay rises slightly with age, and the greater spread among older groups
    indicates more variability — likely due to chronic illnesses or more complex medical conditions.
    """
)


# Boxplot: Total Charges by Age Group
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x="Age Group",
    y="Total Charges",
    order=age_order,
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "red", "markersize": 6}
)
plt.title("Total Charges by Age Group (Niagara County, 2024)")
plt.xlabel("Age Group")
plt.ylabel("Total Charges ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(
    """
    Interpretation:
    Red dots indicate mean values for each age group.
    The mean is higher than the median in most groups, showing that
    hospital charges are right-skewed due to a few very high-cost outliers.
    """
)



# -------------------------------
# 6. SUMMARY 
# -------------------------------
print("\n SUMMARY:\n")
print("""\n In Niagara County’s 2024 inpatient dataset, hospitalization 
      patterns clearly vary by age group. Younger patients experience
      the shortest stays and lowest hospital charges, while middle-aged
      and older adults incur significantly higher costs and longer 
      hospitalizations. The presence of right-skewed distributions,
      shown by higher means than medians, indicates that a small number 
      of very expensive cases disproportionately influence average costs.
      Overall, these findings highlight the strong impact of patient age 
      on healthcare utilization and emphasize the importance of 
      age-specific resource planning in hospital management.""")































# ==============END================

