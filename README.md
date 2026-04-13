# ❤️ CardioRisk AI

AI-powered web application for predicting heart disease risk using Machine Learning.

---

## 🚀 Overview

CardioRisk AI is an end-to-end machine learning project that predicts the likelihood of heart disease based on patient medical data.

The system uses advanced ML models and provides real-time predictions through an interactive web interface.

---

## 🧠 Features

- 🔥 High-performance XGBoost model
- 📊 Real-time prediction with probability
- 🎯 Risk classification (Low / Medium / High)
- 📈 Feature importance visualization
- 💡 Smart medical recommendations
- 🎨 Modern UI built with Streamlit

---

## 📊 Model Performance

| Model | Accuracy |
|------|---------|
| Logistic Regression | 86% |
| Naive Bayes | 85% |
| KNN | 88% |
| Decision Tree | 94% |
| Random Forest | 92% |
| XGBoost | **92.6%** ✅ |

> ⚠️ Note: XGBoost was selected as the final model due to its high recall and generalization ability.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas & NumPy
- Plotly

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/cardiorisk-ai.git
cd cardiorisk-ai

pip install -r requirements.txt
streamlit run app.py
