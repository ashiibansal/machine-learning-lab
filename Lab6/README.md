# Lab 6 – Wine Classification using Decision Tree

## 📌 Aim
To implement the **Decision Tree classification algorithm** on the Wine dataset to classify wine samples based on their chemical properties.

---

## 📂 Dataset Description
- **Dataset Name:** Wine Dataset  
- **Source:** UCI Machine Learning Repository  
- **Total Samples:** 178  
- **Total Features:** 13  
- **Target Variable:** `Class` (Wine type: 1, 2, 3)

Each row represents a wine sample and each column represents a chemical attribute such as Alcohol, Malic Acid, Ash, and Proline.

---

## 🧠 Type of Learning
- **Learning Type:** Supervised Learning  
- **Problem Type:** Multiclass Classification  

The dataset is supervised because each wine sample has a known class label.

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

### Steps Performed
- Dataset overview using `info()` and `describe()`
- Class distribution visualization using **countplot**
- Feature relationships using **pairplot**
- Outlier detection using **boxplots**
- Feature correlation analysis using **heatmap**

EDA helps identify patterns and relationships between features.

---

## ⚙️ Data Preprocessing

### 1️⃣ Feature and Target Separation
- **Features (X):** Chemical properties of wine
- **Target (y):** Wine Class

### 2️⃣ Train-Test Split
The dataset was split into:
- **Training Set:** 80%
- **Testing Set:** 20%

This helps evaluate the model's performance on unseen data.

---

## 🌳 Decision Tree Classification

Decision Trees classify data by splitting features into branches based on decision rules.
The algorithm chooses the best features to split the dataset using measures such as **Gini Index or Entropy**.

---

## 📊 Model Evaluation

The trained model was evaluated using the following metrics:

### Accuracy
Measures the proportion of correctly classified samples.

### Confusion Matrix
Shows the number of correct and incorrect predictions for each class.

### Classification Report
Provides detailed performance metrics including:
- Precision
- Recall
- F1-score
- Support

---

## 🔍 Observations
- The Wine dataset is suitable for multiclass classification.
- Decision Trees can effectively classify wine samples using chemical features.
- The model generates interpretable rules that explain classification decisions.
- Important features such as **Alcohol and Proline** play a significant role in classification.

---

## ✅ Conclusion
In this experiment, a Decision Tree classifier was applied to the Wine dataset to predict wine classes based on chemical attributes. After performing exploratory data analysis and preprocessing, the Decision Tree model was trained and evaluated. The model successfully classified wine samples and provided interpretable decision rules.

---

## 📚 References
- UCI Machine Learning Repository – Wine Dataset  
- Scikit-learn Documentation  
- Matplotlib and Seaborn Documentation  
