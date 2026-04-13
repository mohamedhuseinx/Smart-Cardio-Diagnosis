import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# ===== Load =====
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ===== Config =====
st.set_page_config(page_title="Heart AI", page_icon="❤️", layout="wide")

# ===== CSS =====
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.6);
}
.big-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===== Header =====
st.markdown("<div class='big-title'>❤️ AI Heart Disease Analyzer</div>", unsafe_allow_html=True)
st.write("### 👨‍⚕️ Smart Medical Prediction System")

# ===== Layout =====
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain", [0,1,2,3])
    trestbps = st.number_input("Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("High Sugar", ["No", "Yes"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    restecg = st.selectbox("ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Pain", ["No", "Yes"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Vessels", [0,1,2,3,4])
    thal = st.selectbox("Thal", [0,1,2,3])
    st.markdown("</div>", unsafe_allow_html=True)

# Convert
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# ===== Prediction =====
st.markdown("---")

if st.button("🚀 Analyze Now"):

    with st.spinner("Analyzing patient data..."):
        data = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])

        data_scaled = scaler.transform(data)
        prob = model.predict_proba(data_scaled)[0][1]

    st.markdown("## 🎯 Prediction Result")

    # ===== Result =====
    if prob > 0.7:
        st.error(f"🚨 High Risk ({prob:.2f})")
    elif prob > 0.4:
        st.warning(f"⚠️ Medium Risk ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk ({prob:.2f})")

    st.progress(int(prob * 100))

    # ===== Gauge Chart =====
    fig = px.pie(
        names=["Risk", "Safe"],
        values=[prob, 1-prob],
        title="Risk Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===== Feature Importance =====
    st.markdown("### 🔍 Feature Importance")

    try:
        importance = model.feature_importances_
        features = ["age","sex","cp","trestbps","chol","fbs","restecg",
                    "thalach","exang","oldpeak","slope","ca","thal"]

        df_imp = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig2 = px.bar(df_imp, x="Importance", y="Feature", orientation='h')
        st.plotly_chart(fig2, use_container_width=True)

    except:
        st.info("Feature importance not available.")

    # ===== Advice =====
    st.markdown("### 💡 Medical Advice")

    if prob > 0.7:
        st.write("Consult a cardiologist immediately.")
    elif prob > 0.4:
        st.write("Adopt a healthier lifestyle.")
    else:
        st.write("Maintain your healthy habits.")

# ===== Footer =====
st.markdown("---")
st.markdown("👨‍💻 Developed by Mohamed Hussein")