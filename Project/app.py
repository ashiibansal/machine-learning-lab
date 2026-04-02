import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0f;
    color: #e8e4dd;
}

.block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 50%, #1a0808 100%);
    border: 1px solid rgba(220,38,38,0.25);
    border-radius: 24px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(220,38,38,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ef4444;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    color: #faf5f0;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero-sub {
    color: #9b8f88;
    font-size: 0.95rem;
    font-weight: 400;
    max-width: 800px;
    line-height: 1.6;
}

.card {
    background: #16141a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #ef4444;
    margin-bottom: 0.9rem;
}

.result-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 18px;
    padding: 1.4rem 1.8rem;
    color: #fef2f2;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 1rem 0;
}
.result-low {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 18px;
    padding: 1.4rem 1.8rem;
    color: #f0fdf4;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 1rem 0;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    margin-bottom: 0.3rem;
}

.stat-row { display: flex; gap: 0.8rem; margin: 0.8rem 0; flex-wrap: wrap; }
.stat-box {
    flex: 1;
    min-width: 150px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.1rem;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6570;
    margin-bottom: 0.35rem;
}
.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #faf5f0;
    line-height: 1;
}

.viz-section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #faf5f0;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
}

.section-note {
    color: #9b8f88;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}

section[data-testid="stSidebar"] {
    background-color: #0f0d12 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * { color: #ccc5bb !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stRadio label {
    color: #9b8f88 !important;
    font-size: 0.85rem !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #dc2626, #ef4444) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
}
div.stButton > button:hover { opacity: 0.88 !important; }

.report-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    color: #9b8f88;
    font-size: 0.88rem;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#16141a',
    'axes.facecolor':    '#1e1b24',
    'axes.edgecolor':    '#2e2b36',
    'axes.labelcolor':   '#c4bdb6',
    'axes.titlecolor':   '#faf5f0',
    'xtick.color':       '#7a7280',
    'ytick.color':       '#7a7280',
    'text.color':        '#c4bdb6',
    'grid.color':        '#2e2b36',
    'grid.linestyle':    '--',
    'grid.alpha':        0.6,
    'font.family':       'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

FG = '#faf5f0'
BLUE = '#3b82f6'

MODEL_COLORS = {
    "KNN": "#ef4444",
    "XGBoost": "#f97316",
    "CatBoost": "#f59e0b",
    "Random Forest": "#14b8a6",
    "Gradient Boosting": "#22c55e",
    "Decision Tree": "#84cc16",
    "AdaBoost": "#06b6d4",
    "Logistic Regression": "#3b82f6",
    "Naive Bayes": "#8b5cf6",
    "SVM": "#ec4899",
}

DISPLAY_ORDER = [
    "KNN",
    "XGBoost",
    "CatBoost",
    "Random Forest",
    "Gradient Boosting",
    "Decision Tree",
    "AdaBoost",
    "Logistic Regression",
    "Naive Bayes",
    "SVM",
]

def clean_feature_names(feature_names):
    cleaned = []
    for name in feature_names:
        name = name.replace("num__", "")
        name = name.replace("cat__", "")
        name = name.replace("cat_nom__", "")
        name = name.replace("chest pain type_", "CP=")
        name = name.replace("ST slope_", "Slope=")
        name = name.replace("binary__exercise angina", "Ex. Angina")
        name = name.replace("exercise angina", "Ex. Angina")
        name = name.replace("max heart rate", "Max HR")
        name = name.replace("oldpeak", "Oldpeak")
        cleaned.append(name)
    return cleaned

def build_metrics(y_true, y_pred, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": report,
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc(fpr, tpr),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "y_test": y_true,
    }

def make_leaderboard(metrics):
    rows = []
    for model_name in DISPLAY_ORDER:
        r = metrics[model_name]["report"]
        rows.append({
            "Model": model_name,
            "Accuracy": metrics[model_name]["accuracy"],
            "Precision": r["macro avg"]["precision"],
            "Recall": r["macro avg"]["recall"],
            "F1": r["macro avg"]["f1-score"],
            "AUC": metrics[model_name]["auc"],
        })

    df = pd.DataFrame(rows).sort_values(
        ["Accuracy", "AUC", "F1"], ascending=False
    ).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df

def render_confusion_matrices(selected_models, metrics):
    chunk_size = 3
    for start in range(0, len(selected_models), chunk_size):
        chunk = selected_models[start:start + chunk_size]
        cols = st.columns(len(chunk), gap="medium")
        for col, mname in zip(cols, chunk):
            with col:
                cm_data = metrics[mname]["cm"]
                fig, ax = plt.subplots(figsize=(4, 3.4))
                ax.imshow(cm_data, cmap="Oranges", vmin=0)
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["No CHD", "CHD"], fontsize=10)
                ax.set_yticklabels(["No CHD", "CHD"], fontsize=10)
                ax.set_xlabel("Predicted", fontsize=10, labelpad=6)
                ax.set_ylabel("Actual", fontsize=10, labelpad=6)
                ax.set_title(
                    f"{mname}\nAcc {metrics[mname]['accuracy']*100:.1f}%",
                    fontsize=11, fontweight='700', pad=10
                )

                for r in range(2):
                    for c_idx in range(2):
                        v = cm_data[r, c_idx]
                        ax.text(
                            c_idx, r, str(v),
                            ha='center', va='center',
                            fontsize=20, fontweight='700',
                            color='white' if v > cm_data.max() * 0.45 else FG
                        )

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

@st.cache_resource
def load_all_models():
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

    # Notebook-aligned cleaning
    df["cholesterol"] = df["cholesterol"].replace(0, np.nan)
    df["cholesterol"] = df["cholesterol"].fillna(df["cholesterol"].median())

    # Notebook-aligned outlier capping
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower, upper)

    # Notebook-aligned selected feature subset
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

    # Notebook-aligned train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    num_features = ["oldpeak", "max heart rate"]
    cat_nominal = ["chest pain type", "ST slope"]
    binary_features = ["exercise angina"]

    processed_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat_nom", OneHotEncoder(handle_unknown="ignore"), cat_nominal),
            ("binary", "passthrough", binary_features),
        ]
    )

    raw_preprocessor = FunctionTransformer(func=None, validate=False)

    # Notebook-derived / notebook-matching final configurations
    model_specs = {
        "Logistic Regression": {
            "preprocessor": processed_preprocessor,
            "estimator": LogisticRegression(
                C=0.1,
                max_iter=100,
                penalty="l2",
                solver="lbfgs"
            )
        },
        "Naive Bayes": {
            "preprocessor": raw_preprocessor,
            "estimator": GaussianNB()
        },
        "KNN": {
            "preprocessor": processed_preprocessor,
            "estimator": KNeighborsClassifier(
                n_neighbors=24,
                weights="distance",
                metric="euclidean"
            )
        },
        "SVM": {
            "preprocessor": processed_preprocessor,
            "estimator": SVC(
                C=5,
                gamma="scale",
                kernel="rbf",
                probability=True
            )
        },
        # Chosen to match the notebook's final comparison accuracy panel
        "Decision Tree": {
            "preprocessor": processed_preprocessor,
            "estimator": DecisionTreeClassifier(
                max_depth=4
            )
        },
        "Random Forest": {
            "preprocessor": processed_preprocessor,
            "estimator": RandomForestClassifier(
                n_estimators=100,
                criterion="entropy",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        },
        "XGBoost": {
            "preprocessor": processed_preprocessor,
            "estimator": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
                verbosity=0
            )
        },
        "CatBoost": {
            "preprocessor": processed_preprocessor,
            "estimator": CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                verbose=0,
                random_state=42
            )
        },
        "AdaBoost": {
            "preprocessor": processed_preprocessor,
            "estimator": AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            )
        },
        "Gradient Boosting": {
            "preprocessor": processed_preprocessor,
            "estimator": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
        },
    }

    pipelines = {}
    metrics = {}

    for model_name in DISPLAY_ORDER:
        spec = model_specs[model_name]
        pipe = Pipeline([
            ("preprocessor", clone(spec["preprocessor"])),
            ("classifier", spec["estimator"])
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        pipelines[model_name] = pipe
        metrics[model_name] = build_metrics(y_test, y_pred, y_proba)

    leaderboard = make_leaderboard(metrics)
    return pipelines, metrics, leaderboard

with st.spinner("Training all 10 notebook models… (first load only)"):
    pipelines, metrics, leaderboard = load_all_models()

top_5_models = leaderboard["Model"].head(5).tolist()
top_3_models = leaderboard["Model"].head(3).tolist()
all_models_ranked = leaderboard["Model"].tolist()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Clinical Risk Screening · ML Dashboard</div>
    <div class="hero-title">Heart Disease Prediction System</div>
    <div class="hero-sub">
        This interface exposes all 10 notebook-based models in the backend and dashboard,
        while keeping the primary view focused on the best-performing shortlist. The full
        comparison page is preserved for report screenshots and academic evaluation.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("### 🌟 Primary View")

    primary_model = st.radio(
        "Primary model (Top 5 only)",
        top_5_models,
        index=0
    )

    plot_models = st.multiselect(
        "Models to overlay in plots",
        options=all_models_ranked,
        default=top_3_models
    )
    if not plot_models:
        plot_models = top_3_models

    st.markdown("---")
    st.markdown("### 🩺 Patient Parameters")

    cp = st.selectbox(
        "Chest Pain Type",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-Anginal Pain",
            4: "Asymptomatic"
        }[x]
    )

    slope = st.selectbox(
        "ST Slope",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x]
    )

    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 10.0, 1.0, 0.1)

    exang = st.selectbox(
        "Exercise-Induced Angina",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    thalach = st.slider("Max Heart Rate", 60, 220, 150)

    st.markdown("---")
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True)

# ── Primary Layout ────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1.05], gap="large")

with col_l:
    st.markdown('<div class="card"><div class="card-title">Patient Summary</div>', unsafe_allow_html=True)

    st.dataframe(
        pd.DataFrame({
            "Feature": [
                "Chest Pain Type",
                "ST Slope",
                "Oldpeak",
                "Exercise Angina",
                "Max Heart Rate"
            ],
            "Value": [
                {
                    1: "Typical Angina",
                    2: "Atypical Angina",
                    3: "Non-Anginal Pain",
                    4: "Asymptomatic"
                }[cp],
                {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[slope],
                oldpeak,
                "Yes" if exang else "No",
                thalach
            ]
        }),
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        input_df = pd.DataFrame(
            [[cp, slope, oldpeak, exang, thalach]],
            columns=[
                "chest pain type",
                "ST slope",
                "oldpeak",
                "exercise angina",
                "max heart rate"
            ]
        )

        pred = int(pipelines[primary_model].predict(input_df)[0])
        proba = pipelines[primary_model].predict_proba(input_df)[0]
        conf = float(np.max(proba) * 100)

        if pred == 1:
            st.markdown(f"""<div class="result-high">
                <div class="result-title">⚠️ CHD likely</div>
                Primary model: {primary_model} &nbsp;·&nbsp; Model confidence {conf:.1f}%
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-low">
                <div class="result-title">✅ No CHD detected</div>
                Primary model: {primary_model} &nbsp;·&nbsp; Model confidence {conf:.1f}%
            </div>""", unsafe_allow_html=True)

        st.progress(conf / 100)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-label">P(No Disease)</div>
                <div class="stat-value">{proba[0]*100:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">P(Disease)</div>
                <div class="stat-value" style="color:#ef4444">{proba[1]*100:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Primary Model</div>
                <div class="stat-value" style="font-size:1.05rem">{primary_model}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        prediction_rows = []
        for model_name in all_models_ranked:
            model_pred = int(pipelines[model_name].predict(input_df)[0])
            model_proba = pipelines[model_name].predict_proba(input_df)[0]
            prediction_rows.append({
                "Model": model_name,
                "Rank": int(leaderboard.loc[leaderboard["Model"] == model_name, "Rank"].iloc[0]),
                "P(No Disease)": f"{model_proba[0]*100:.1f}%",
                "P(Disease)": f"{model_proba[1]*100:.1f}%",
                "Predicted Class": "Disease" if model_pred == 1 else "No Disease",
                "Confidence": f"{max(model_proba)*100:.1f}%"
            })

        pred_df = pd.DataFrame(prediction_rows).sort_values("Rank")

        st.markdown("### Primary comparison — Top 5 models")
        st.dataframe(
            pred_df[pred_df["Model"].isin(top_5_models)],
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Show all 10 model predictions for this patient"):
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

with col_r:
    st.markdown('<div class="card"><div class="card-title">Primary View — Top 5 Models</div>', unsafe_allow_html=True)

    top5_df = leaderboard.head(5).copy()
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    bars = ax.bar(
        top5_df["Model"],
        top5_df["Accuracy"] * 100,
        color=[MODEL_COLORS[m] for m in top5_df["Model"]],
        edgecolor='none'
    )
    for bar, value in zip(bars, top5_df["Accuracy"] * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=FG,
            fontweight="600"
        )
    ax.set_title("Top 5 Accuracy Ranking", fontsize=11, fontweight="700", pad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(80, 95)
    ax.tick_params(axis='x', labelsize=8, rotation=12)
    ax.grid(axis='y')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    top5_display = top5_df.copy()
    top5_display["Accuracy"] = (top5_display["Accuracy"] * 100).map(lambda x: f"{x:.1f}%")
    top5_display["AUC"] = top5_display["AUC"].map(lambda x: f"{x:.3f}")
    top5_display["F1"] = (top5_display["F1"] * 100).map(lambda x: f"{x:.1f}%")
    st.dataframe(
        top5_display[["Rank", "Model", "Accuracy", "AUC", "F1"]],
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="viz-section-title">📊 Performance Visualisations</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-note">Selected overlay models: <strong style="color:#faf5f0;">{", ".join(plot_models)}</strong></div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔷 Confusion Matrix",
    "📈 ROC Curve",
    "🎯 Precision · Recall · F1",
    "📄 Full 10-Model Comparison"
])

# ── TAB 1: Confusion matrices for selected models ─────────────────────────────
with tab1:
    st.markdown(
        '<div class="section-note">Choose the models to visualise from the sidebar. '
        'Confusion matrices are generated from the same held-out test split used for the leaderboard.</div>',
        unsafe_allow_html=True
    )
    render_confusion_matrices(plot_models, metrics)

# ── TAB 2: ROC overlay for selected models ────────────────────────────────────
with tab2:
    roc_left, roc_right = st.columns([1.7, 1], gap="large")

    with roc_left:
        fig, ax = plt.subplots(figsize=(7.4, 5.2))

        for mname in plot_models:
            color = MODEL_COLORS[mname]
            lw = 3.0 if mname == primary_model else 2.0
            alpha = 1.0 if mname == primary_model else 0.75

            ax.plot(
                metrics[mname]["fpr"],
                metrics[mname]["tpr"],
                color=color,
                linewidth=lw,
                alpha=alpha,
                label=f"{mname} (AUC = {metrics[mname]['auc']:.3f})"
            )

        ax.plot([0, 1], [0, 1], linestyle='--', color='#3e3b46', linewidth=1.2, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=11, labelpad=8)
        ax.set_ylabel("True Positive Rate", fontsize=11, labelpad=8)
        ax.set_title("ROC Overlay — User-selected models", fontsize=12, fontweight='700', pad=12)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.15, facecolor='#16141a', edgecolor='#2e2b36')
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with roc_right:
        st.markdown("""
        <div class="report-box">
        <strong style="color:#faf5f0;">Why this matters:</strong><br><br>
        The ROC view is now user-selectable, so your report screenshots can show either
        the best 3 models or the full shortlisted comparison without cluttering the figure.
        This keeps the interface academically faithful to the notebook while staying readable.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for mname in plot_models:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
            background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.06);
            border-radius:12px;padding:0.6rem 0.9rem;margin-bottom:0.5rem;'>
                <span style='font-weight:600;color:#faf5f0;font-size:0.9rem;'>{mname}</span>
                <span style='font-weight:700;color:{MODEL_COLORS[mname]};font-size:1.05rem;'>
                AUC {metrics[mname]["auc"]:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 3: metrics ────────────────────────────────────────────────────────────
with tab3:
    pr_l, pr_r = st.columns([1, 1], gap="large")

    with pr_l:
        report = metrics[primary_model]["report"]
        cats = [
            "Precision\nClass 0", "Recall\nClass 0", "F1\nClass 0",
            "Precision\nClass 1", "Recall\nClass 1", "F1\nClass 1"
        ]
        vals = [
            report["0"]["precision"], report["0"]["recall"], report["0"]["f1-score"],
            report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"]
        ]
        bar_colors = [BLUE] * 3 + [MODEL_COLORS.get(primary_model, "#ef4444")] * 3

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        bars = ax.bar(cats, [v * 100 for v in vals], color=bar_colors, width=0.6, edgecolor='none')
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v*100:.1f}%",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='600',
                color=FG
            )
        ax.set_ylim(0, 108)
        ax.set_title(f"{primary_model} — Per-class Metrics", fontsize=11, fontweight='700', pad=10)
        ax.set_ylabel("Score (%)", fontsize=10)
        ax.grid(axis='y')
        ax.legend(
            handles=[
                mpatches.Patch(color=BLUE, label='Class 0 — No CHD'),
                mpatches.Patch(color=MODEL_COLORS.get(primary_model, "#ef4444"), label='Class 1 — CHD')
            ],
            fontsize=9,
            framealpha=0.15,
            facecolor='#16141a',
            edgecolor='#2e2b36'
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with pr_r:
        metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        x = np.arange(len(metric_labels))
        width = 0.8 / max(len(plot_models), 1)

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for i, mname in enumerate(plot_models):
            r = metrics[mname]["report"]
            values = [
                metrics[mname]["accuracy"],
                r["macro avg"]["precision"],
                r["macro avg"]["recall"],
                r["macro avg"]["f1-score"],
                metrics[mname]["auc"]
            ]
            offset = (i - (len(plot_models) - 1) / 2) * width
            ax.bar(
                x + offset,
                [v * 100 for v in values],
                width=width,
                color=MODEL_COLORS[mname],
                alpha=0.9,
                label=mname,
                edgecolor='none'
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(75, 100)
        ax.set_ylabel("Score (%)", fontsize=10)
        ax.set_title("Macro-metric Overlay — User-selected models", fontsize=11, fontweight='700', pad=10)
        ax.legend(fontsize=8, framealpha=0.15, facecolor='#16141a', edgecolor='#2e2b36')
        ax.grid(axis='y')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ── TAB 4: full 10-model comparison page ─────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="report-box">
    <strong style="color:#faf5f0;">Report screenshot page:</strong><br><br>
    This section is designed to mirror the full comparative analysis format:
    a complete 10-model accuracy chart, an all-model leaderboard, and a selectable
    feature-importance panel for the tree / boosting models.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    full_l, full_r = st.columns([1.2, 1], gap="large")

    with full_l:
        fig, ax = plt.subplots(figsize=(9.8, 4.8))
        ordered_df = leaderboard.copy()
        bars = ax.bar(
            ordered_df["Model"],
            ordered_df["Accuracy"] * 100,
            color=[MODEL_COLORS[m] for m in ordered_df["Model"]],
            edgecolor='none'
        )
        for bar, value in zip(bars, ordered_df["Accuracy"] * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=FG,
                fontweight='600'
            )
        ax.set_title("Model Comparison (Accuracy) — All 10 Models", fontsize=12, fontweight='700', pad=12)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(78, 93)
        ax.tick_params(axis='x', labelsize=8, rotation=18)
        ax.grid(axis='y')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Probability distributions for selected models
        fig, axes = plt.subplots(1, len(plot_models), figsize=(4.2 * len(plot_models), 3.6), squeeze=False)
        axes = axes[0]

        for ax, mname in zip(axes, plot_models):
            yp = metrics[mname]["y_proba"]
            yt = metrics[mname]["y_test"].values

            ax.hist(yp[yt == 0], bins=18, color=BLUE, alpha=0.7, label='No CHD', edgecolor='none')
            ax.hist(yp[yt == 1], bins=18, color=MODEL_COLORS[mname], alpha=0.7, label='CHD', edgecolor='none')
            ax.set_title(mname, fontsize=10, fontweight='700')
            ax.set_xlabel("Predicted P(CHD)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(axis='y')

        axes[0].legend(fontsize=8, framealpha=0.2, facecolor='#16141a', edgecolor='none')
        fig.suptitle("Prediction Probability Distributions — Selected Models", fontsize=10, fontweight='700', color=FG, y=1.03)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with full_r:
        leaderboard_display = leaderboard.copy()
        leaderboard_display["Accuracy"] = (leaderboard_display["Accuracy"] * 100).map(lambda x: f"{x:.1f}%")
        leaderboard_display["Precision"] = (leaderboard_display["Precision"] * 100).map(lambda x: f"{x:.1f}%")
        leaderboard_display["Recall"] = (leaderboard_display["Recall"] * 100).map(lambda x: f"{x:.1f}%")
        leaderboard_display["F1"] = (leaderboard_display["F1"] * 100).map(lambda x: f"{x:.1f}%")
        leaderboard_display["AUC"] = leaderboard_display["AUC"].map(lambda x: f"{x:.3f}")

        st.markdown("""
        <div style='font-size:0.72rem;font-weight:600;letter-spacing:0.14em;
        text-transform:uppercase;color:#ef4444;margin-bottom:0.8rem;'>
        Expanded Comparison — All 10 Models
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            leaderboard_display[["Rank", "Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]],
            use_container_width=True,
            hide_index=True
        )

        importance_candidates = [
            m for m in DISPLAY_ORDER
            if hasattr(pipelines[m].named_steps["classifier"], "feature_importances_")
        ]
        importance_model = st.selectbox(
            "Feature importance model",
            importance_candidates,
            index=importance_candidates.index("Random Forest") if "Random Forest" in importance_candidates else 0
        )

        st.markdown(f"""
        <div style='font-size:0.72rem;font-weight:600;letter-spacing:0.14em;
        text-transform:uppercase;color:{MODEL_COLORS[importance_model]};margin:1.2rem 0 0.6rem;'>
        Feature Importance — {importance_model}
        </div>
        """, unsafe_allow_html=True)

        model_pipe = pipelines[importance_model]
        model_obj = model_pipe.named_steps["classifier"]
        preprocessor = model_pipe.named_steps["preprocessor"]

        fi = model_obj.feature_importances_
        feat_names = preprocessor.get_feature_names_out()
        cleaned_feat_names = clean_feature_names(feat_names)

        fi_df = pd.DataFrame({
            "Feature": cleaned_feat_names[:len(fi)],
            "Importance": fi
        }).sort_values("Importance", ascending=True).tail(8)

        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        bars = ax.barh(
            fi_df["Feature"],
            fi_df["Importance"] * 100,
            color=MODEL_COLORS[importance_model],
            alpha=0.88,
            edgecolor='none'
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 0.25,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%",
                va='center',
                fontsize=8,
                color=FG
            )
        ax.set_xlabel("Importance (%)", fontsize=9)
        ax.set_title(f"Top Features — {importance_model}", fontsize=10, fontweight='700', pad=8)
        ax.grid(axis='x')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#3e3b46;font-size:0.82rem;
margin-top:2.5rem;padding-top:1.2rem;border-top:1px solid #1e1b24;'>
    Heart Disease Prediction · 10-model notebook-aligned comparison ·
    Logistic Regression · Naive Bayes · KNN · SVM · Decision Tree · Random Forest ·
    XGBoost · CatBoost · AdaBoost · Gradient Boosting
</div>
""", unsafe_allow_html=True)
