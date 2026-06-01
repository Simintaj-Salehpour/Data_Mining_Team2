Data Mining Team 2: Niagara County Inpatient Outcomes Analysis
Project Overview
This project examines how patient demographics, insurance type, admission characteristics, clinical severity, and age relate to hospital resource utilization in Niagara County, New York. The analysis uses the Hospital Inpatient Discharges (SPARCS De-Identified), 2024 dataset and focuses on inpatient discharges from Niagara County.

The project applies exploratory data analysis, regression modeling, classification, and machine learning techniques to understand patterns in:

Length of Stay (LOS)
Total Charges
Total Costs
Insurance-based charge differences
Heart failure charges by age group
Age-related hospitalization patterns
Authors
Simintaj Salehpour
Nazish Atta
Alex Biuckians
Dataset
Dataset: Hospital Inpatient Discharges (SPARCS De-Identified), 2024
Geographic focus: Niagara County, New York
Subset size: Approximately 6,912 inpatient discharges
Number of variables: 34

The dataset includes information related to:

Patient demographics: age group, gender, race, ethnicity
Admission characteristics: admission type, emergency department indicator, discharge status
Clinical information: diagnosis codes, APR DRG, APR MDC, severity of illness, risk of mortality
Insurance/payment type: Medicaid, Medicare, Private Insurance, Federal/State/VA, and other payer categories
Outcomes: length of stay, total charges, and total costs
Research Questions
SMART Question 1
What combination of patient demographics, admission type, and comorbidities predicts the lowest hospital charges for Medicaid vs. Private Insurance patients in 2024?

This question compares Medicaid and Private Insurance patients using:

Linear Regression
Logistic Regression for low-charge classification
Random Forest Regression
The goal is to understand which clinical and demographic factors are most associated with lower hospital charges and whether charges are more predictable for one insurance group than another.

SMART Question 2
For Niagara County inpatient discharges in 2024 with a primary diagnosis code starting with Heart Failure, how do median Total Charges vary across age groups? Can a regression model predict total hospital charges for heart failure with RMSE below $5,000 using age group, length of stay, and payer type?

This question focuses specifically on heart failure cases and evaluates:

Median total charges by age group
Linear Regression performance
Random Forest Regression performance
Whether the model meets the RMSE target of less than $5,000
SMART Question 3
How does patient age affect hospitalization factors like Length of Stay and Total Charges across various age groups in 2024?

This question explores whether older patients experience longer stays and higher hospital charges by using:

Descriptive statistics
Pearson correlation
Scatterplots with regression trend lines
Boxplots by age group
Methods
Data Cleaning and Preparation
The raw SPARCS dataset was cleaned and prepared through the following steps:

Converted Length of Stay, Total Charges, and Total Costs into numeric values.
Removed rows with missing values in key variables.
Reconstructed payer categories into broader insurance groups.
Collapsed rare categorical levels into an Other category.
Created separate subsets for Medicaid and Private Insurance patients.
Filtered Niagara County records for county-level analysis.
Created a heart failure subset using the relevant diagnosis category/code.
Engineered ChargePerDay to reduce the dominance of Length of Stay in cost prediction.
Exploratory Data Analysis
EDA was used to understand distributions and relationships across variables. The analysis included:

Age group, gender, race, and ethnicity distributions
Admission type and payment typology distributions
Length of Stay and Total Charges distributions
Correlation heatmap of numeric features
Boxplots of LOS and charges by age group
Heart failure charge distributions by age group
Modeling Approach
Linear Regression
Linear Regression was used to estimate total hospital charges from demographic, admission, and clinical predictors.

Key result:

Medicaid Linear Regression R²: 0.803
Private Insurance Linear Regression R²: 0.703
This suggests that Medicaid charges were more closely explained by observed clinical and demographic variables, while Private Insurance charges showed more unexplained variation.

Logistic Regression
Logistic Regression was used to classify encounters into low-charge vs. high-charge groups.

Key result:

Medicaid accuracy: 0.9096
Private Insurance accuracy: 0.8375
This indicates stronger low-charge classification performance for Medicaid patients compared with Private Insurance patients.

Random Forest Regression
Random Forest models were used to identify important predictors of daily hospital cost. Because Length of Stay is strongly correlated with Total Charges, the project used:

ChargePerDay = Total Charges / Length of Stay
Important predictors included:

Diagnosis category
APR Severity of Illness
Age group
Emergency department indicator
Admission type
Heart Failure Regression Models
For the heart failure analysis, Linear Regression and Random Forest Regression were tested using:

Age Group
Length of Stay
Payment Typology 1
Model performance:

Model	Cross-Validated RMSE	Test RMSE
Linear Regression	$13,001.33	$15,767.34
Random Forest Regression	$14,857.98	$16,269.92
Neither model met the target RMSE of less than $5,000. This suggests that age group, length of stay, and payer type alone were not enough to explain the high variability in heart failure hospital charges.

Key Findings
Insurance-Based Findings
Medicaid patients showed more stable and predictable cost patterns than Private Insurance patients. Across Linear Regression, Logistic Regression, and Random Forest models, Medicaid charges were more strongly aligned with clinical and demographic variables.

Private Insurance charges showed greater variability, which may reflect factors not captured in the dataset, such as negotiated payment rates, plan-specific differences, or administrative billing variation.

Heart Failure Findings
For heart failure patients:

There were 380 heart failure cases in the Niagara County subset.
Median total charges were similar across age groups, around $17,500.
Older age groups had higher means and larger standard deviations, suggesting more high-cost outlier cases.
Regression models could not predict charges with RMSE below $5,000.
Median heart failure charges by age group:

Age Group	Count	Median Total Charges	Mean Total Charges	Standard Deviation
30-49	12	$17,310.08	$16,253.84	$6,446.14
50-69	127	$17,758.21	$26,372.43	$25,543.60
70 or Older	241	$17,734.11	$22,909.55	$20,008.66
Age and Hospitalization Findings
The analysis found a strong positive relationship between Length of Stay and Total Charges.

Key result:

Pearson correlation between Length of Stay and Total Charges: r = 0.78
This indicates that longer hospital stays are strongly associated with higher hospital charges.

The findings also suggest that older adults tend to experience longer hospitalizations and higher treatment costs, especially among middle-aged and older patient groups.

Project Files
File	Description
Hospital_Inpatient_Discharges_2024.csv	Main SPARCS inpatient discharge dataset used for the analysis
Project_Team2_Data_Mining.py	Main Python script containing data cleaning, EDA, and analyses for all three SMART questions
Project_Team2_Smart_Question1.py	Python script focused on SMART Question 1 modeling
Question2.py	Python script for the heart failure analysis
Atta_nazish.py	Python script for age, LOS, and charge analysis
Team2_Summary_Report.docx	Final written project report
Proposal_Team2.docx	Project proposal document
Data_Mining_Team2.pptx	Presentation file
Data_Mining_Team2_Slides.pptx	Slide deck
Data_Mining_Team2_Presentation	Presentation-related project file
Technologies Used
Python
pandas
NumPy
matplotlib
seaborn
scikit-learn
statsmodels
Altair
How to Run the Project
1. Clone the Repository
git clone https://github.com/Simintaj-Salehpour/Data_Mining_Team2.git
cd Data_Mining_Team2
2. Create a Virtual Environment
python -m venv .venv
Activate it:

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
3. Install Required Packages
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels altair
4. Run the Main Script
Make sure the dataset file is in the same folder as the Python scripts:

python Project_Team2_Data_Mining.py
You can also run individual analysis scripts:

python Project_Team2_Smart_Question1.py
python Question2.py
python Atta_nazish.py
Results Summary
This project shows that hospital charges and length of stay are influenced by a combination of insurance type, age, diagnosis category, admission type, and illness severity.

The strongest overall findings are:

Medicaid charges were more predictable than Private Insurance charges.
Clinical severity and diagnosis category were among the most important predictors of hospital cost.
Heart failure median charges were similar across older age groups, but high-cost outliers increased the mean and variability.
Length of Stay and Total Charges had a strong positive correlation.
Older patients generally experienced longer stays and higher hospital charges.
Limitations
This project is limited to Niagara County inpatient discharges from 2024, so the findings may not generalize to other counties, states, or years.

The dataset is administrative and billing-focused. It does not include detailed clinical information such as lab results, medications, imaging findings, follow-up outcomes, or patient-level social determinants of health. These missing variables may explain additional variation in hospital charges and length of stay.

The charge data is highly right-skewed, meaning that a small number of very high-cost cases can strongly affect averages and model performance.

Future Work
Future improvements could include:

Adding clinical severity variables to the heart failure prediction model
Comparing Niagara County with other New York counties
Testing additional machine learning models such as Gradient Boosting or XGBoost
Applying log transformation to Total Charges to reduce skewness
Using feature selection and hyperparameter tuning
Expanding the analysis to multiple years
Studying cost disparities by race, ethnicity, payer type, and admission source
Conclusion
This project demonstrates how data mining can support healthcare cost analysis and hospital resource planning. By combining descriptive statistics, visualization, regression, classification, and machine learning, the analysis provides insight into how demographics, insurance type, clinical severity, and age influence hospital utilization in Niagara County.

The findings can help support more data-driven decision-making for hospital budgeting, staffing, patient flow management, and equitable care delivery.
