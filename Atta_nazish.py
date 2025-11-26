#%%
# ------------------------------------------
# Question #3 Analysis Script
# How does patient age (measured by Age Group)
# affect Length of Stay and Total Charges in 2024 (Niagara County)?
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. LOAD DATA
# -------------------------------
raw_df = pd.read_csv("Hospital_Inpatient_Discharges_2024.csv")
df = raw_df.copy()
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

# %%


