# Credit Card Default Prediction using Machine Learning

## Project Overview
Perform data cleaning, preprocessing, and visualization to prepare datasets for machine learning tasks. 

This project focuses on predicting whether a customer will default on their credit card payment in the next month using machine learning techniques.  
The dataset contains demographic information, credit limits, bill amounts, payment history, and repayment behavior of customers.

---

## 🔄 Preprocessing Steps
The following preprocessing steps were applied to prepare the data for machine learning:

1. Fixed incorrect header row and cleaned column names  
2. Converted all features to numeric format  
3. Handled missing values using **median imputation**  
4. Removed irrelevant features (ID column)  
5. Analyzed class distribution (imbalanced dataset)  
6. Applied **feature scaling using StandardScaler**  
7. Split the dataset into training and testing sets (80:20 split with stratification)

---

## 📊 Exploratory Data Analysis (EDA)
Data visualization was performed to better understand the dataset:

- Class distribution plot to analyze imbalance  
- Boxplots to detect outliers in numerical features  
- Feature vs target analysis (Age, Credit Limit, Payment Status)  
- Correlation heatmap to identify highly correlated features  
- Pairplot (on selected features) to analyze relationships and class separation  

---
