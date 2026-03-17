# Telco Customer Churn Predictor 📉📈

A machine learning web application that predicts the likelihood of a telecommunications customer canceling their service based on demographic, contract, and behavioral data.

**🔴 Live Dashboard:** [churn-predictor-telco.streamlit.app](https://churn-predictor-telco.streamlit.app/)

---

## 🛠️ Tech Stack
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud

## 📊 Model Architecture & Performance
The core prediction engine is an **XGBoost Classifier** trained on the Kaggle Telco Customer Churn dataset (7,043 records).

* **Preprocessing Pipeline:** Automated via `Scikit-Learn Pipelines` (One-Hot Encoding for categorical features, Standard Scaling for numerical metrics like tenure and charges).
* **Hyperparameter Tuning:** Utilized `RandomizedSearchCV` to strictly regularize the model and prevent overfitting.
* **Results:** Successfully closed the training-test generalization gap from 7.6% down to **2.23%**, resulting in a highly robust and stable **79.1% testing accuracy** on unseen data.

## 📂 Repository Structure
* `app.py`: The Streamlit web application and UI logic.
* `churn_model.pkl`: The serialized, pre-trained XGBoost inference model.
* `research.ipynb`: Complete Jupyter Notebook containing EDA, feature engineering, and model training/tuning.
* `requirements.txt`: Minimal dependency list required for cloud deployment.
* [`WA_Fn-UseC_-Telco-Customer-Churn.csv1`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): The raw historical dataset.

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Avijit-1234/churn-predictor-telco.git](https://github.com/Avijit-1234/churn-predictor-telco.git)
   cd churn-predictor-telco
   ```

2. **Create and activate a virtual environment:**
   Windows
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   macOS/Linux
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Boot up the local server:**
   ```bash
   streamlit run app.py
   ```
