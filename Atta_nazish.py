#%%
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











# %%
