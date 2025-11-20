#%%
# ------------------------------------------
# Question #3 Analysis Script
# How does patient age affect Length of Stay
# and Total Charges across age groups in 2024?
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. LOAD DATA
# -------------------------------
# Assumes the CSV is in the same folder as this .py file
df = pd.read_csv("Hospital_Inpatient_Discharges_2024.csv")


# %%
