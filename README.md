# 🎓 Student Performance Predictor

A machine learning-based web application that predicts a student's overall academic performance grade (A–F) using key academic activity metrics such as attendance, assignment completion, test scores, practical scores, and exam scores. Built with XGBoost and deployed as an interactive dark-themed web app using Streamlit.

🔗 **Live Demo:** https://student-performance-system-ml-project-app.streamlit.app/

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

---

## 🔍 Overview

Identifying at-risk students early allows educators to intervene before performance issues become critical. This project builds a supervised machine learning classification model that predicts a student's final performance grade based on measurable academic engagement metrics, removing the need to wait for final exam results to flag underperformance.

The final model is served through a custom dark-themed Streamlit interface, allowing users to adjust input metrics via sliders and instantly receive a predicted grade along with a model confidence score and visual score breakdown.

---

## ✨ Features

- Interactive, dark-themed UI with a custom card-based layout
- Real-time grade prediction (A–F) with color-coded grade badges
- Model confidence score displayed alongside predictions
- Visual score breakdown chart for each input metric
- Dataset overview with live summary statistics
- End-to-end ML pipeline (scaling + classification) built with Scikit-learn and XGBoost

---

## 🛠 Tech Stack

| Category            | Tools/Libraries                          |
|---------------------|-------------------------------------------|
| Language            | Python                                    |
| Data Processing     | Pandas, NumPy                             |
| Visualization (EDA) | Matplotlib, Seaborn                       |
| Machine Learning    | Scikit-learn, XGBoost                     |
| Model Serialization | Joblib                                    |
| Web Framework       | Streamlit                                 |
| Development         | Jupyter Notebook / Google Colab           |

---

## 📂 Project Structure

```
Student-Performance-Prediction/
│
├── app.py                              # Streamlit web application (dark theme UI)
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Pinned Python version for stable deployment
├── .gitignore
├── README.md
│
├── data/
│   └── clean_student.csv                # Cleaned dataset used for dataset insights
│
├── model/
│   └── student_performance_pipeline.pkl # Trained scaling + classification pipeline
│
└── notebook/
    └── student_performance_model.ipynb  # Full EDA, preprocessing & model training notebook
```

---

## 📊 Dataset

The dataset consists of **5,000 student records** with the following features used for prediction:

| Feature                  | Description                                  |
|----------------------------|-----------------------------------------------|
| `Attendance`              | Attendance percentage                        |
| `Assignment Completion`   | Percentage of assignments completed           |
| `Test Score`              | Test score (out of 25)                        |
| `Practical Score`         | Practical/lab score (out of 25)               |
| `Exam Score`              | Final exam score (out of 50)                  |
| `Performance` (target)    | Overall grade — A, B, C, D, E, or F            |

Additional demographic and lifestyle columns (age, gender, marital status, family background, etc.) were present in the raw dataset but were excluded from modeling, as they showed limited relevance to academic performance prediction and reduced model interpretability.

---

## ⚙️ Methodology

1. **Data Cleaning** — Verified the dataset for missing values and duplicates (none found). Removed 14 non-academic columns to focus on performance-relevant features.
2. **Exploratory Data Analysis (EDA)** — Analyzed distributions, correlations, and relationships between academic metrics and final performance using histograms, box plots, violin plots, and a correlation heatmap.
3. **Target Encoding** — Ordinally encoded the `Performance` column (F=0 → A=5) to reflect its natural grade ordering.
4. **Feature Scaling** — Standardized all numerical features using `StandardScaler`.
5. **Model Comparison** — Trained and evaluated four classification algorithms:
   - Random Forest Classifier
   - Bagging Classifier
   - Extra Trees Classifier
   - XGBoost Classifier
6. **Pipeline Construction** — Combined scaling and the best-performing classifier into a single Scikit-learn `Pipeline` to ensure consistent preprocessing between training and inference.
7. **Deployment** — Serialized the final pipeline with Joblib and deployed via a custom Streamlit interface.

---

## 📈 Model Performance

| Model                    | Accuracy | Precision | Recall | F1-Score |
|----------------------------|----------|-----------|--------|----------|
| Random Forest Classifier  | 93.9%    | 0.940     | 0.939  | 0.939    |
| Bagging Classifier        | 93.0%    | 0.931     | 0.930  | 0.930    |
| Extra Trees Classifier    | 94.0%    | 0.941     | 0.940  | 0.940    |
| **XGBoost Classifier**    | **94.4%**| **0.945** | **0.944**| **0.944**|

✅ **XGBoost Classifier** was selected as the final model based on its highest accuracy and consistently strong precision, recall, and F1-score across all metrics.

---

## 🚀 Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/samichohan/Student-Performance-Prediction.git
cd Student-Performance-Prediction

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`) in your browser. Adjust the sliders for attendance, assignment completion, test score, practical score, and exam score, then click **Predict Performance** to view the predicted grade and confidence score.

---

## 🔮 Future Improvements

- Add SHAP-based feature importance to explain individual predictions
- Hyperparameter tuning via GridSearchCV/Optuna for further accuracy gains
- Incorporate additional behavioral/engagement features with careful bias analysis
- Add historical trend tracking for repeat student assessments
- Containerize the app with Docker for consistent, platform-independent deployment

---

## 👤 Author

**Sami Chohan**
GitHub: [@samichohan](https://github.com/samichohan)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
