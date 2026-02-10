# Lab 5 – Logistic Regression on Wine Dataset

## 📌 Aim
To apply descriptive statistics, preprocessing, feature scaling, feature selection, and Logistic Regression on the Wine dataset and evaluate the classification model.

---

## 📂 Dataset Description
- **Dataset Name:** Wine Dataset  
- **Source:** UCI Machine Learning Repository  
- **Number of Instances:** 178  
- **Number of Features:** 13  
- **Target Variable:** `Class` (Wine Class: 1, 2, 3)

---

## 🧠 Type of Learning
- **Learning Type:** Supervised Learning  
- **Problem Type:** Multiclass Classification  

The dataset is supervised because class labels are known.

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

### 1. Data Loading & Understanding
- Loaded the Wine dataset using Pandas
- Checked dataset structure using `info()` and `describe()`
- Verified absence of missing values

---

### 2. Descriptive Statistical Analysis
- Calculated mean, median, standard deviation
- Analyzed minimum, maximum, quartiles, and percentiles
- Studied feature distributions and variability

---

### 3. Exploratory Data Analysis (EDA)
- Visualized class distribution using count plot
- Compared feature values across classes using boxplots
- Analyzed feature relationships using correlation heatmap
- Used pairplot for visual feature relationship analysis

---

### 4. Preprocessing
- Separated features and target variable
- Removed low-impact and redundant features
- Verified data consistency

---

### 5. Feature Scaling
- Applied **StandardScaler** to normalize feature values
- Scaling was necessary due to different numerical ranges

---

### 6. Feature Selection
- Used correlation analysis to detect redundancy
- Applied coefficient-based feature selection using Logistic Regression
- Dropped low-impact features such as:
  - Magnesium  
  - Nonflavanoid Phenols  
  - Total Phenols  

---

### 7. Train–Test Split
- Split data into training (80%) and testing (20%) sets
- Used stratified sampling to maintain class balance

---

### 8. Logistic Regression Model
- Applied Logistic Regression with **One-vs-Rest (OvR)** strategy
- Trained the model on selected and scaled features

---

### 9. Model Evaluation
- Evaluated model using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

---

## 📊 Observations
- The dataset is suitable for multiclass classification
- Feature scaling improved model performance
- Feature selection reduced redundancy and overfitting
- Logistic Regression performed well using OvR strategy
- No encoding techniques were required as all features were numerical

---

## ✅ Conclusion
This experiment demonstrated the complete machine learning pipeline, including data understanding, preprocessing, feature scaling, feature selection, and model evaluation. Logistic Regression successfully classified wine samples into their respective classes with good generalization performance.
