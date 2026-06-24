# Credit Risk Prediction App 

An End-to-End Machine Learning project that predicts the credit risk of bank customers (Good vs. Bad Credit Risk) based on their demographic and financial profile. The project includes interactive data analysis, machine learning modeling using XGBoost, and a web deployment using Streamlit.

---

##  Features
- **Data Cleaning & Preprocessing:** - *Handling Missing Values (Row Dropping):* Rows containing missing or unrecorded inputs (`NaN`) within critical financial metrics—specifically `Saving accounts` and `Checking account`—were dropped directly from the baseline data matrix to eliminate noise and prevent prediction bias.
- **Exploratory Data Analysis (EDA):** Insightful data visualization with Seaborn/Matplotlib analyzing correlations, distributions, and customer behaviors.
- **Advanced Machine Learning:** Hyperparameter tuning via `GridSearchCV` across multiple models (Decision Tree, Random Forest, Extra Trees, and XGBoost).
- **Interactive Web UI:** Built with Streamlit, allowing users to input financial indicators and get real-time credit risk assessments.

---

##  Tech Stack & Libraries
- **Frontend/Deployment:** Streamlit
- **Data Handling:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, Joblib
- **Visualization:** Matplotlib, Seaborn

---

##  Model Performance
After testing and tuning multiple classifiers, **XGBoost** emerged as the champion model with the highest cross-validation accuracy:

| Model | Test Accuracy |
| :--- | :--- |
| Decision Tree | ~58.1% |
| Random Forest | ~61.9% |
| **XGBoost (Best Model)** | **~67.6%** |

---

##  Project Structure
```text
├── dataset/
│   └── german_credit_data.csv       # Original dataset
├── encoders/
│   ├── sex_encoder.pkl              # Saved LabelEncoders
│   ├── housing_encoder.pkl
│   ├── saving_accounts_encoder.pkl
│   └── checking_account_encoder.pkl
├── xgb_credit_risk_model.pkl        # Trained XGBoost model file
├── credit_risk_notebook.ipynb       # Jupyter Notebook for EDA & Model Training
├── app.py                           # Streamlit Web Application
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
