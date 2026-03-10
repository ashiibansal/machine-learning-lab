# Lab 7 – Wine Classification using Support Vector Machine (SVM)

## 📌 Aim
To implement the **Support Vector Machine (SVM)** algorithm on the Wine dataset to classify wine samples based on their chemical properties, and improve model performance using **GridSearchCV for hyperparameter tuning**.

---

## 📂 Dataset Description
- **Dataset Name:** Wine Dataset
- **Source:** UCI Machine Learning Repository
- **Total Samples:** 178
- **Total Features:** 13
- **Target Variable:** `Class` (Wine type: 1, 2, 3)

Each row represents a wine sample, and each column represents a chemical attribute such as Alcohol, Malic Acid, Ash, Proline, etc.

---

## 🧠 Type of Learning
- **Learning Type:** Supervised Learning  
- **Problem Type:** Multiclass Classification  

The dataset is supervised because the correct class label for each wine sample is already known.

---

## 🛠 Tools and Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔎 Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset before applying machine learning models.

### Steps performed:
- Dataset overview using `info()` and `describe()`
- Class distribution visualization using **countplot**
- Feature relationship visualization using **pairplot**
- Outlier detection using **boxplots**
- Feature correlation analysis using **heatmap**

EDA helps identify patterns, relationships, and potential issues in the dataset.

---

## ⚙️ Data Preprocessing

### 1️⃣ Feature and Target Separation
- **Features (X):** Chemical attributes of wine
- **Target (y):** Wine Class

### 2️⃣ Feature Scaling
Since SVM is sensitive to feature scale, **StandardScaler** was applied to normalize feature values.

### 3️⃣ Train-Test Split
The dataset was divided into:
- **Training Set:** 80%
- **Testing Set:** 20%

This allows evaluation of model performance on unseen data.

---

## ⚡ Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm used for classification tasks.

SVM works by finding the **optimal hyperplane** that separates classes with the **maximum margin**.

For multiclass classification, SVM uses the **One-vs-Rest (OvR)** strategy internally.

---

## 🔧 Hyperparameter Tuning using GridSearchCV

Choosing the correct parameters for SVM is important for model performance.

**GridSearchCV** was used to test multiple combinations of parameters and select the best-performing model.
GridSearchCV performs:
- Multiple model training iterations
- Cross-validation
- Selection of optimal parameters

---

## 📊 Model Evaluation

The trained model was evaluated using the following metrics:

### Accuracy
Measures the proportion of correct predictions.

### Confusion Matrix
Shows the number of correct and incorrect predictions for each class.

### Classification Report
Provides detailed metrics including:
- Precision
- Recall
- F1-score
- Support

---

## 🔍 Observations

- The Wine dataset is suitable for multiclass classification.
- Feature scaling significantly improves SVM performance.
- SVM successfully separates wine classes based on chemical attributes.
- Hyperparameter tuning using GridSearchCV improves model accuracy and reliability.
- The model generalizes well on unseen test data.

---

## ✅ Conclusion

In this experiment, the Support Vector Machine algorithm was applied to the Wine dataset for classification. After performing exploratory data analysis, preprocessing, and feature scaling, the SVM model was trained and optimized using GridSearchCV. The final model achieved good classification performance, demonstrating the effectiveness of SVM for multiclass classification tasks.

---

## 📚 References
- UCI Machine Learning Repository – Wine Dataset
- Scikit-learn Documentation
- Seaborn & Matplotlib Documentation
