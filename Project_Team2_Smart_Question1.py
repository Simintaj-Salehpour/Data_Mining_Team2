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
# 5. SMART Question 1: What combination of patient demographics, admission type, and comorbidities predicts the lowest hospital charges for Medicaid vs. Private Insurance patients in 2024?
# ==============================

print(df.columns.tolist())
print("Initial count:", len(df))

# 2. Convert numeric columns
numeric_cols = ['Length of Stay', 'Total Charges', 'Total Costs']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Identify Payment Typology columns
payment_cols = [c for c in df.columns if c.startswith("Payment Typology")]
print("Payment columns:", payment_cols)


# Assign Insurance Type (matching your EXACT R patterns)
def assign_insurance(row):
    # find payment columns that are "on" (value = 1)
    pay_cols_on = [col for col in payment_cols if row[col] == 1]

    # combine them into a single string
    combined = " ".join(pay_cols_on).lower()

    # R patterns converted to python
    medicaid_pattern = re.compile(r"medicaid|federal/state/local/va")
    private_pattern  = re.compile(r"blue cross|private health insurance|private|miscellaneous/other")

    if medicaid_pattern.search(combined):
        return "Medicaid"
    if private_pattern.search(combined):
        return "Private"
    return "Other"

df['Insurance'] = df.apply(assign_insurance, axis=1)
print("Insurance counts:\n", df['Insurance'].value_counts())
print("After Insurance filtering:", len(df))


# Encode categorical variables
df['Emergency Department Indicator'] = df['Emergency Department Indicator'].map({'Y':1, 'N':0})

df['APR Severity of Illness Description'] = (
    df['APR Severity of Illness Description']
      .astype('category')
      .cat.codes
)

df['CCSR Diagnosis Description'] = (
    df['CCSR Diagnosis Description']
      .astype('category')
      .cat.codes
)


# Log-transform Total Charges (increases R² significantly)
df['Total Charges Log'] = np.log1p(df['Total Charges'])
target = 'Total Charges Log'


# Select final model features
model_features = [
    'Length of Stay',

    # Age groups
    'Age Group_18-29', 'Age Group_30-49',
    'Age Group_50-69', 'Age Group_70 or Older',

    # Gender
    'Gender_M',

    # Race
    'Race_Multi-racial', 'Race_Other Race', 'Race_White',

    # Type of Admission
    'Type of Admission_Emergency', 'Type of Admission_Newborn',
    'Type of Admission_Not Available', 'Type of Admission_Urgent',

    # Other important fields
    'Emergency Department Indicator',
    'APR Severity of Illness Description',
    'CCSR Diagnosis Description'
]


# Build final clean dataset
hospital_final = df[model_features + ['Insurance', target]].dropna()
print("Final dataset for modeling:", len(hospital_final))


# Split dataset by Insurance
medicaid_df = hospital_final[hospital_final['Insurance'] == 'Medicaid']
private_df  = hospital_final[hospital_final['Insurance'] == 'Private']

print("Medicaid count:", len(medicaid_df))
print("Private count:", len(private_df))

medicaid_train, medicaid_test = train_test_split(medicaid_df, train_size=0.7, random_state=123)
private_train,  private_test  = train_test_split(private_df,  train_size=0.7, random_state=123)

# Tuned Random Forest Models (High R²)
rf_medicaid = RandomForestRegressor(
    n_estimators=1000,
    max_features='sqrt',
    min_samples_split=5,
    random_state=123
)

rf_private = RandomForestRegressor(
    n_estimators=1000,
    max_features='sqrt',
    min_samples_split=5,
    random_state=123
)

# 11. Train models
rf_medicaid.fit(medicaid_train[model_features], medicaid_train[target])
rf_private.fit(private_train[model_features], private_train[target])


# Evaluate models
medicaid_pred = rf_medicaid.predict(medicaid_test[model_features])
private_pred  = rf_private.predict(private_test[model_features])

print("\n MODEL PERFORMANCE ")
print("Random Forest R² (Medicaid):", r2_score(medicaid_test[target], medicaid_pred))
print("Random Forest R² (Private):", r2_score(private_test[target], private_pred))


# Top 5 feature importances
def print_top_features(model, features, title):
    importances = pd.Series(model.feature_importances_, index=features)
    print(f"\nTop 5 Important Features ({title}):")
    print(importances.sort_values(ascending=False).head(5))

print_top_features(rf_medicaid, model_features, "Medicaid")
print_top_features(rf_private, model_features, "Private")



#%% 
# ==============================
# 6. SMART Question 2: For all Hospital Inpatient Discharges in Niagara County, NY in 2024 with a Primary Diagnosis Code starting with Heart Failure, how does the median Total Charges vary across all defined Age Groups in the Niagara County subset of the data? Can a regression model predict a patient’s total hospital charges for heart failure with a root mean squared error (RMSE) below $5,000, using age group, length of stay, and payer type as predictors?
# ==============================
















#%% 
# ==============================
# 7. SMART Question 3: How does patient age affect hospitalization factors like Length of Stay and Total Charges across various age groups in 2024?
# ==============================



















# ==============END================

