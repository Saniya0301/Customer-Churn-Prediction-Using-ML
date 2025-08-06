# Customer-Churn-Prediction-Using-ML

---

🚀 Overview


This project builds a complete churn prediction pipeline using Python and machine learning. We use a real-world dataset (Kaggle Telco Customer Churn) to train, evaluate, and deploy models that predict whether a customer is likely to churn.

📋 Dataset

Publicly available Telco Customer Churn dataset from Kaggle (e.g. IBM/Telco churn dataset)

Records include demographics, subscribed services, account details, and churn status
  

**Key Features:**
- Real-world dataset from the telecom sector
- Complete machine learning workflow:
  - Data preprocessing
  - Exploratory Data Analysis (EDA)
  - Model training and evaluation
  - Accuracy comparison
- Uses logistic regression, decision tree, random forest, and SVM

---

## 🧠 Algorithms Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

---

## 📚 Dataset

- Dataset: Telco Customer Churn
- Source: [Kaggle Telco Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- Features: Gender, SeniorCitizen, Partner, Tenure, InternetService, MonthlyCharges, etc.

---

## 🧰 Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn
- scikit-learn (sklearn)

---

🛠 Methodology
Exploratory Data Analysis (EDA): Analyze missing data, discover correlations, visualize distributions.

Preprocessing:

Handle missing values

Encode categorical variables

Scale numerical features

Manage class imbalance via SMOTE or SMOTE‑ENN

Modeling:

Train classification algorithms: Logistic Regression, Random Forest, XGBoost

Tune hyperparameters (e.g. via GridSearchCV or Optuna) 
Evaluation:

Metrics: accuracy, precision, recall, F1‑score, ROC‑AUC

Focus on high recall to reduce false negatives (missing churners)

## 🧪 Project Workflow

1. **Importing the dataset**
2. **Data cleaning & preprocessing**
   - Handle missing values
   - Label encoding for categorical values
   - Scaling features
3. **Splitting dataset into training and test sets**
4. **Training models**
5. **Evaluating models using accuracy scores**
6. **Comparing performances**

---

## 📈 Model Accuracy

| Model              | Accuracy (Approx.) |
|-------------------|---------------------|
| Logistic Regression | ~80%               |
| Decision Tree       | ~78%               |
| Random Forest       | ~83%               |
| SVM                 | ~81%               |

> 📌 *Note: Accuracy may vary slightly based on preprocessing and random state.*

---
📌 Future Improvements
Add hyperparameter tuning

Deploy the model using Streamlit or Flask

Visualize SHAP values for model explainability

📬 Feel free to raise issues or contribute to the project!

📝 License
This project is open-source and available under the MIT License.

