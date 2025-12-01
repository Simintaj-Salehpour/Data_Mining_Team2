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

# Convert numeric columns safely
numeric_cols = ['Length of Stay', 'Total Charges', 'Total Costs']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
 
payment_cols = [col for col in df.columns if col.startswith("Payment Typology")]

def assign_insurance(row):
    medicaid_pattern = re.compile(r'Medicaid|Federal/State/Local/VA', re.IGNORECASE)
    private_pattern  = re.compile(r'Private|Blue Cross|Miscellaneous', re.IGNORECASE)
    for col in payment_cols:
        val = str(row[col])
        if medicaid_pattern.search(val):
            return 'Medicaid'
        elif private_pattern.search(val):
            return 'Private'
    return 'Other'

df['Insurance'] = df.apply(assign_insurance, axis=1)
df = df[df['Insurance'].isin(['Medicaid','Private'])]

#%% 
# 3. Keep relevant features
target = 'Total Charges'
features = [
    'Length of Stay',
    # Age Group dummies
    'Age Group_18-29', 'Age Group_30-49', 'Age Group_50-69', 'Age Group_70 or Older',
    # Gender dummies
    'Gender_M',
    # Race dummies
    'Race_Multi-racial', 'Race_Other Race', 'Race_White',
    # Type of Admission dummies
    'Type of Admission_Emergency', 'Type of Admission_Newborn',
    'Type of Admission_Not Available', 'Type of Admission_Urgent',
    # Other categorical columns
    'Emergency Department Indicator',
    'APR Severity of Illness Description',
    'CCSR Diagnosis Description'
]

hospital_final = df[features + ['Insurance', target]].dropna()

#%% 
# 4. Split by Insurance type
medicaid_df = hospital_final[hospital_final['Insurance']=='Medicaid']
private_df  = hospital_final[hospital_final['Insurance']=='Private']
#%%
def split_train_test(df, train_ratio=0.7, random_state=123):
    return train_test_split(df, train_size=train_ratio, random_state=random_state)

medicaid_train, medicaid_test = split_train_test(medicaid_df)
private_train, private_test   = split_train_test(private_df)




#%% 
# ==============================
# 6. SMART Question 2: For all Hospital Inpatient Discharges in Niagara County, NY in 2024 with a Primary Diagnosis Code starting with Heart Failure, how does the median Total Charges vary across all defined Age Groups in the Niagara County subset of the data? Can a regression model predict a patient’s total hospital charges for heart failure with a root mean squared error (RMSE) below $5,000, using age group, length of stay, and payer type as predictors?
# ==============================
















#%% 
# ==============================
# 7. SMART Question 3: How does patient age affect hospitalization factors like Length of Stay and Total Charges across various age groups in 2024?
# ==============================



















# ==============END================

