# 🫀 Heart Disease Risk Prediction Engine & Clinical Stress-Testing

An end-to-end Machine Learning pipeline and interactive web deployment built to evaluate cardiovascular disease risk. Trained on the **UCI Cleveland Heart Disease dataset**, this project goes beyond standard validation metrics by stress-testing the model against real-world clinical case studies (ranging from adult congenital heart disease and end-stage congestive failure to pediatric acute rheumatic fever and viral myopericarditis).

---

## 📸 Interactive Web App (Streamlit)

![App Demo](https://github.com/hamza-hk7440/heart-disease-prediction/blob/master/demo.mp4) <!-- Replace with your actual screenshot/GIF path -->

The pipeline is wrapped in a lightweight, interactive **Streamlit** application that accepts real-time patient parameters, applies fitted standardization scalers, and computes diagnostic probabilities.

---

## 📊 Key Features & Performance Summary

* **Dataset:** UCI Cleveland Heart Disease Dataset (303 patient records)
* **Classifier:** Logistic Regression with L2 Regularization
* **Model Benchmark:**
  * **Test Accuracy:** `80.43%`
  * **ROC-AUC Score:** `0.90`
* **Feature Engineering:** Custom interaction term `age_bps_ratio` ($age \times resting\_bp$) to capture multiplicative risk in hypertensive aging demographics.
* **Preprocessor:** `StandardScaler` fitted on training splits to prevent data leakage.

---

## 🛠️ Project Architecture & Workflow

1. **Data Preprocessing & Encoding:**
   * One-hot encoding categorical features (`cp`, `restecg`, `slope`).
   * Handling missing/implausible clinical entries.
   * Standardizing numerical variables (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`).

2. **Model Persistence:**
   * Saved model weights: `best_logistic_regression_model.pkl`
   * Saved preprocessor: `scaler.pkl`

3. **Inference Pipeline:**
   * Aligns arbitrary input feature dictionaries with the exact training one-hot schema ($x_{train}$).
   * Computes log-odds scores and maps them through the sigmoid function $P(Y=1) = \frac{1}{1 + e^{-z}}$.

---

## 🩺 Real-World Clinical Case Validation

To evaluate out-of-distribution performance and boundary limits, the model was tested against published clinical case studies:

| Case Description | Clinical Profile | Model Output Probability | Diagnostic Analysis |
| :--- | :--- | :---: | :--- |
| **78yo Male (Ethiopian Case Study)** | Single Atrium anomaly, CHF, AFib, Bifascicular Block, BP 150/75, HR 50 bpm | **`92.00%` (High Risk)** | **True Positive Match:** Correctly flags severe, end-stage cardiovascular failure in an elderly male profile. |
| **44yo Female (NEJM Case)** | Acute COVID-19 Myopericarditis & Cardiogenic Shock (Normal Coronary Arteries) | **`24.17%` (Low CAD Risk)** | **Domain Boundary:** Model accurately reflects low chronic CAD probability despite acute viral inflammatory state. |
| **8.5yo Female (Case 4)** | Acute Rheumatic Fever, Mitral Regurgitation, Active GAS Culture | **`24.80%` (Low CAD Risk)** | Correctly differentiates pediatric autoimmune heart disease from adult ischemic heart disease. |

---

## 💻 Installation & Local Usage

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
python -m venv .venv
```

Activate environment:
* **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
* **Linux/macOS:** `source .venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 📁 Repository Directory Structure

```text
├── heart_disease_analysis.ipynb          # EDA, Model Training, & Case Evaluation
├── app.py                                # Streamlit Web Application Interface
├── heart_disease_dataset.csv             # Processed Dataset
└── README.md                             # Project Documentation
```

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Model Serialization:** Joblib
* **Web UI:** Streamlit
