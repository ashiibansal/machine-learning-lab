import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-card {
        background: linear-gradient(135deg, #ff4b4b, #ff7b7b);
        padding: 2rem;
        border-radius: 22px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: white;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        color: white;
    }

    .info-card {
        background: white;
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        border: 1px solid #eef2f7;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.6rem;
    }

    .small-text {
        color: #475569;
        font-size: 0.95rem;
    }

    .prediction-heading {
        color: #0f172a;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .result-box {
        padding: 1.4rem;
        border-radius: 18px;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .low-risk {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        box-shadow: 0 8px 24px rgba(34,197,94,0.25);
    }

    .high-risk {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        box-shadow: 0 8px 24px rgba(239,68,68,0.25);
    }

    .prob-card {
        background: white;
        border: 1px solid #eef2f7;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
    }

    .prob-label {
        color: #475569;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .prob-value {
        color: #0f172a;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 2rem;
    }

    /* Keep dataframe/card headings readable */
    div[data-testid="stDataFrame"] * {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div class="hero-title">❤️ Heart Disease Prediction Dashboard</div>
    <div class="hero-subtitle">
        Clinical risk screening using a K-Nearest Neighbors model trained on the Heart Statlog Cleveland-Hungary dataset.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Train model on startup ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

    # Handle zero cholesterol values
    df["cholesterol"] = df["cholesterol"].replace(0, np.nan)
    df["cholesterol"] = df["cholesterol"].fillna(df["cholesterol"].median())

    # Outlier capping using IQR
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    selected_cols = [
        "chest pain type",
        "ST slope",
        "oldpeak",
        "exercise angina",
        "max heart rate",
        "target"
    ]
    df = df[selected_cols].copy()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["oldpeak", "max heart rate"]),
        ("cat_nom", OneHotEncoder(handle_unknown="ignore"), ["chest pain type", "ST slope"]),
        ("binary", "passthrough", ["exercise angina"])
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean",
            weights="uniform"
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

pipeline = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Inputs")
    st.markdown("Adjust the clinical parameters below to estimate heart disease risk.")

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-Anginal Pain",
            3: "Asymptomatic"
        }[x]
    )

    slope = st.selectbox(
        "ST Slope",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    thalach = st.slider(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=220,
        value=150
    )

    predict_btn = st.button("🔍 Predict Risk", use_container_width=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    st.markdown("""
    <div class="info-card">
        <div class="section-title">Patient Summary</div>
        <div class="small-text">
            Review the selected input variables before running the prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    summary_df = pd.DataFrame({
        "Feature": [
            "Chest Pain Type",
            "ST Slope",
            "Oldpeak",
            "Exercise-Induced Angina",
            "Max Heart Rate"
        ],
        "Value": [
            {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-Anginal Pain",
                3: "Asymptomatic"
            }[cp],
            {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[slope],
            oldpeak,
            "Yes" if exang == 1 else "No",
            thalach
        ]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with right_col:
    st.markdown("""
    <div class="info-card">
        <div class="section-title">Model Information</div>
        <div class="small-text">
            Algorithm: K-Nearest Neighbors (KNN)<br>
            Input Features: 5 selected clinical variables<br>
            Dataset: Heart Statlog Cleveland-Hungary
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Prediction Section ────────────────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([[cp, slope, oldpeak, exang, thalach]], columns=[
        "chest pain type",
        "ST slope",
        "oldpeak",
        "exercise angina",
        "max heart rate"
    ])

    prediction = pipeline.predict(input_df)
    proba = pipeline.predict_proba(input_df)[0]
    confidence = float(np.max(proba) * 100)

    st.markdown('<div class="prediction-heading">Prediction Result</div>', unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown(
            f"""
            <div class="result-box high-risk">
                ⚠️ High Risk of Heart Disease<br>
                Model confidence: {confidence:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-box low-risk">
                ✅ Low Risk of Heart Disease<br>
                Model confidence: {confidence:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    st.progress(confidence / 100)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div class="prob-card">
                <div class="prob-label">Probability: No Disease</div>
                <div class="prob-value">{proba[0] * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="prob-card">
                <div class="prob-label">Probability: Disease</div>
                <div class="prob-value">{proba[1] * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("""
    <div class="info-card">
        <div class="section-title">Clinical Note</div>
        <div class="small-text">
            This prediction is intended for educational and screening purposes only.
            It does not replace formal clinical diagnosis, ECG interpretation, imaging,
            or physician evaluation.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-note">
        Trained on the Heart Statlog Cleveland-Hungary dataset · KNN classifier · Streamlit interface
    </div>
    """,
    unsafe_allow_html=True
)