# Lab 04 – Supervised Learning using USA Housing Dataset

##  Aim
To apply supervised learning techniques on the USA Housing dataset and analyze the relationship between housing features and house prices using Linear Regression.

---

## 📂 Dataset Description
- **Dataset Name:** USA Housing Dataset
- **Type:** Supervised Dataset
- **Number of Instances:** 5000
- **Number of Features:** 5
- **Target Variable:** `Price`

### Features:
- Avg. Area Income  
- Avg. Area House Age  
- Avg. Area Number of Rooms  
- Avg. Area Number of Bedrooms  
- Area Population  

---

## 🧠 Type of Learning
- **Learning Type:** Supervised Learning  
- **Problem Type:** Regression  

The dataset is supervised because the target variable (house price) is known.

---

## 🛠 Tools & Libraries Used
- Python
- Jupyter Notebook / Google Colab
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔄 Steps Performed

### 1. Data Loading
- Loaded the USA Housing dataset using Pandas
- Inspected dataset structure and columns

### 2. Data Understanding
- Identified features and target variable
- Verified data types and checked for missing values

### 3. Descriptive Statistical Analysis
- Calculated mean, median, mode
- Computed minimum, maximum, quartiles, percentiles
- Analyzed standard deviation and variance
- Studied correlation and covariance between variables

### 4. Exploratory Data Analysis (EDA)
- Visualized feature distributions
- Plotted correlation heatmap
- Identified relationships between features and target variable

### 5. Data Preprocessing
- Separated features and target variable
- Applied feature scaling where required
- Split data into training and testing sets

### 6. Model Application
- Applied Linear Regression model
- Predicted house prices based on input features

---

## 📊 Observations
- House price shows strong correlation with average income and number of rooms
- Some features are moderately correlated with each other
- Data shows linear relationships suitable for regression
- No major missing values observed

---

## 📈 Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## ✅ Conclusion
This lab demonstrates the application of supervised learning using Linear Regression on the USA Housing dataset. Descriptive statistical analysis and data visualization helped understand feature relationships and prepare the data for model training. Linear Regression was found to be suitable for predicting house prices.

---

## 📚 References
- Kaggle – USA Housing Dataset  
- Scikit-learn Documentation  
