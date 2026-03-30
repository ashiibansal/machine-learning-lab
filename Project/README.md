# Heart Disease Prediction using Machine Learning

## 📌 Overview
This project predicts the likelihood of heart disease using multiple machine learning algorithms. A comparative analysis of 10 classification models was performed to determine the most effective approach.

The system uses a merged dataset (Statlog, Cleveland, Hungary) and applies preprocessing, feature selection, and hyperparameter tuning to achieve high accuracy.

---

## 🎯 Objectives
- Predict heart disease (binary classification)
- Compare multiple ML models
- Identify the best-performing algorithm
- Build an efficient and interpretable system

---

## 📊 Dataset
- **Source:** UCI Heart Disease Dataset (Cleveland, Statlog, Hungary)
- **Records:** 1190
- **Features:** 12 (reduced to 5 after feature selection)

### ✅ Selected Features
- Chest Pain Type  
- ST Slope  
- Oldpeak  
- Exercise Induced Angina  
- Max Heart Rate  

---

## ⚙️ Methodology

### 🔹 Data Preprocessing
- Missing value handling (median imputation)
- Outlier capping using IQR method
- Correlation-based feature selection

### 🔹 Feature Engineering
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer pipeline

### 🔹 Train-Test Split
- 75% Training / 25% Testing  
- `random_state = 42`

### 🔹 Model Training
- Hyperparameter tuning using GridSearchCV (5-fold cross-validation)

---

## 🤖 Models Used
- Logistic Regression  
- Naive Bayes  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- XGBoost  
- CatBoost  
- AdaBoost  
- Gradient Boosting  

---

## 📈 Results

| Model | Accuracy |
|------|--------|
| KNN | **90.3%** |
| XGBoost | 87.2% |
| CatBoost | 87.2% |
| Random Forest | 86.6% |
| Gradient Boosting | 86.2% |
| Decision Tree | 84.2% |
| AdaBoost | 84.2% |
| Logistic Regression | 83.5% |
| Naive Bayes | 82.9% |
| SVM | 82.9% |

---

## 🔍 Key Insights
- KNN achieved the highest accuracy (90.3%)
- Ensemble methods performed consistently well
- Dataset shows non-linear patterns
- Preprocessing significantly improved performance

---

## 📊 Visualizations
- Correlation Heatmap  
- Model Comparison Graph  
- Confusion Matrix  
- ROC Curve  
- Feature Importance  

---

## 🚀 Technologies Used
- Python  
- NumPy, Pandas  
- Scikit-learn  
- XGBoost, CatBoost  
- Matplotlib, Seaborn

---

## 🖥️ Deployment (Streamlit)

### ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure
ML_Project/

│── app.py

│── model.pkl

│── dataset.csv

│── notebook.ipynb

│── README.md
