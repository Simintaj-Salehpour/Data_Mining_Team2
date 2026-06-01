
# Examining Demographics, Admission Types, and Clinical Outcomes in New York Inpatients

## Project Overview

This data mining project analyzes inpatient hospital discharges from **Niagara County, New York** using the **Hospital Inpatient Discharges (SPARCS De-Identified), 2024** dataset.

The goal of this project is to examine how patient demographics, insurance type, admission type, clinical severity, and age relate to hospital outcomes such as:

- Length of Stay
- Total Charges
- Total Costs
- Insurance-based cost differences
- Heart failure hospital charges
- Age-related hospitalization patterns

The analysis supports healthcare resource planning, cost forecasting, and data-driven decision-making for hospital operations.

## Team Members

- Simintaj Salehpour
- Nazish Atta
- Alex Biuckians

## Dataset

**Dataset:** Hospital Inpatient Discharges (SPARCS De-Identified), 2024  
**Location:** Niagara County, New York  
**Subset Size:** Approximately 6,912 inpatient discharges  
**Variables:** 34 columns

The dataset includes:

- Patient demographics: age group, gender, race, ethnicity
- Admission details: admission type, emergency department indicator, discharge status
- Clinical information: diagnosis codes, APR DRG, severity of illness, risk of mortality
- Insurance/payment type: Medicaid, Medicare, Private Insurance, and other payer types
- Outcomes: length of stay, total charges, and total costs

## Research Questions

### Question 1

What combination of patient demographics, admission type, and comorbidities predicts the lowest hospital charges for Medicaid vs. Private Insurance patients in 2024?

### Question 2

For Niagara County inpatient discharges in 2024 with a primary diagnosis related to heart failure, how do median total charges vary across age groups? Can a regression model predict total hospital charges for heart failure with RMSE below $5,000 using age group, length of stay, and payer type?

### Question 3

How does patient age affect hospitalization factors such as Length of Stay and Total Charges across different age groups in 2024?

## Methods

The project used the following data mining and statistical methods:

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Descriptive statistics
- Correlation analysis
- Linear Regression
- Logistic Regression
- Random Forest Regression
- Feature importance analysis
- Boxplots, histograms, scatterplots, and heatmaps

## Data Preparation

The raw dataset was cleaned and prepared by:

- Converting Length of Stay, Total Charges, and Total Costs into numeric format
- Removing rows with missing values in key fields
- Grouping insurance categories into broader payer groups
- Collapsing rare categorical values into “Other”
- Creating Medicaid and Private Insurance subsets
- Filtering heart failure cases for Question 2
- Creating Charge per Day to reduce the effect of Length of Stay on cost prediction

## Key Findings

### Insurance and Hospital Charges

Medicaid patients showed more stable and predictable hospital charge patterns than Private Insurance patients.

Linear Regression results:

| Insurance Type | R-squared |
|---|---:|
| Medicaid | 0.803 |
| Private Insurance | 0.703 |

Logistic Regression results:

| Insurance Type | Accuracy |
|---|---:|
| Medicaid | 0.9096 |
| Private Insurance | 0.8375 |

The results suggest that Medicaid charges were more closely related to observed clinical and demographic variables, while Private Insurance charges showed greater unexplained variation.

### Heart Failure Analysis

The heart failure subset included **380 patients**.

Median total charges were similar across age groups, but older age groups showed higher variation and more high-cost cases.

| Age Group | Count | Median Total Charges | Mean Total Charges |
|---|---:|---:|---:|
| 30-49 | 12 | $17,310.08 | $16,253.84 |
| 50-69 | 127 | $17,758.21 | $26,372.43 |
| 70 or Older | 241 | $17,734.11 | $22,909.55 |

Regression models did not meet the target RMSE of less than $5,000.

| Model | Test RMSE |
|---|---:|
| Linear Regression | $15,767.34 |
| Random Forest Regression | $16,269.92 |

This suggests that age group, length of stay, and payer type alone were not enough to accurately predict heart failure charges.

### Age, Length of Stay, and Charges

The analysis found a strong positive relationship between Length of Stay and Total Charges.

```text
Pearson correlation = 0.78
