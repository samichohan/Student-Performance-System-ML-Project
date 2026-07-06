import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM DARK THEME CSS ---------------- #
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #E6E6E6;
    }

    h1, h2, h3 {
        color: #F5F5F5;
        font-family: 'Segoe UI', sans-serif;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #7F5AF0;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9A9CA5;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    .card {
        background-color: #161A23;
        border: 1px solid #262B36;
        border-radius: 14px;
        padding: 1.6rem;
        margin-bottom: 1rem;
    }

    .grade-badge {
        display: inline-block;
        font-size: 3.2rem;
        font-weight: 800;
        padding: 0.6rem 1.8rem;
        border-radius: 16px;
        text-align: center;
    }

    .metric-label {
        color: #9A9CA5;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        color: #F5F5F5;
        font-size: 1.6rem;
        font-weight: 700;
    }

    .stButton > button {
        background-color: #7F5AF0;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #6b46e0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL & DATA ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("model/student_performance_pipeline.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/clean_student.csv")

try:
    model = load_model()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Required file not found: {e}")
    st.info("Make sure 'model/student_performance_pipeline.pkl' and 'data/clean_student.csv' exist.")
    st.stop()

# Grade order used during training (ordinal encoding)
grade_order = ['F', 'E', 'D', 'C', 'B', 'A']

grade_colors = {
    'A': '#2CB67D',
    'B': '#4C9AFF',
    'C': '#F5C242',
    'D': '#F2994A',
    'E': '#EF5DA8',
    'F': '#EF4565',
}

# ---------------- HEADER ---------------- #
st.markdown('<p class="main-title">🎓 Student Performance Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Estimate a student\'s overall performance grade from academic activity metrics.</p>', unsafe_allow_html=True)

# ---------------- INPUT CARD ---------------- #
left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Academic Inputs")

    attendance = st.slider("Attendance (%)", 0, 100, 80)
    assignment = st.slider("Assignment Completion (%)", 0, 100, 80)
    test_score = st.slider("Test Score (out of 25)", 0, 25, 15)
    practical_score = st.slider("Practical Score (out of 25)", 0, 25, 15)
    exam_score = st.slider("Exam Score (out of 50)", 0, 50, 25)

    predict = st.button("🔮 Predict Performance")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Debug Info"):
        import sklearn
        st.write("Scikit-learn:", sklearn.__version__)
        st.write("Joblib:", joblib.__version__)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Prediction Result")

    if predict:
        try:
            input_df = pd.DataFrame([{
                "Attendance": attendance,
                "Assignment Completion": assignment,
                "Test Score": test_score,
                "Practical Score": practical_score,
                "Exam Score": exam_score
            }])

            pred_encoded = model.predict(input_df)[0]
            predicted_grade = grade_order[int(pred_encoded)]
            color = grade_colors.get(predicted_grade, "#7F5AF0")

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                confidence = proba[int(pred_encoded)]

            st.markdown(
                f'<div class="grade-badge" style="background-color:{color}22; '
                f'color:{color}; border: 2px solid {color};">{predicted_grade}</div>',
                unsafe_allow_html=True
            )

            if confidence is not None:
                st.markdown(
                    f'<p class="metric-label" style="margin-top:1rem;">Model Confidence</p>'
                    f'<p class="metric-value">{confidence*100:.1f}%</p>',
                    unsafe_allow_html=True
                )

            st.divider()
            st.markdown("**Score Breakdown**")
            chart_df = pd.DataFrame({
                "Metric": ["Attendance", "Assignment", "Test", "Practical", "Exam"],
                "Score": [attendance, assignment, test_score, practical_score, exam_score]
            }).set_index("Metric")
            st.bar_chart(chart_df, color="#7F5AF0")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Adjust the sliders and click **Predict Performance** to see the result.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DATASET INSIGHTS ---------------- #
st.divider()
st.subheader("📁 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Students", len(df))
c2.metric("Avg. Attendance", f"{df['Attendance'].mean():.1f}%")
c3.metric("Avg. Exam Score", f"{df['Exam Score'].mean():.1f}")
c4.metric("Top Grade Count (A)", int((df["Performance"] == "A").sum()))

with st.expander("View Dataset Sample"):
    st.dataframe(df.head(10))

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Built with Streamlit + XGBoost · Student Performance Prediction System")