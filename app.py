import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Regret Forecast Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Styling (dark/light friendly)
# =========================
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
      .hero{
        border-radius:18px;
        padding:18px 20px;
        background: linear-gradient(90deg, rgba(2,132,199,1) 0%, rgba(99,102,241,1) 50%, rgba(168,85,247,1) 100%);
        color:white;
        box-shadow: 0 10px 28px rgba(0,0,0,0.15);
        margin-bottom:12px;
      }
      .hero h1{ margin:0; font-size:34px; line-height:1.15; }
      .hero p{ margin:8px 0 0 0; opacity:0.95; font-size:14px; }
      .card{
        border-radius:16px;
        padding:16px;
        border:1px solid rgba(120,120,120,0.25);
        background: rgba(255,255,255,0.04);
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }
      .small { font-size: 13px; opacity: 0.85; }
      .good { padding: 10px 12px; border-radius: 12px; background: rgba(34,197,94,0.15); border:1px solid rgba(34,197,94,0.35); }
      .warn { padding: 10px 12px; border-radius: 12px; background: rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.35); }
      .bad  { padding: 10px 12px; border-radius: 12px; background: rgba(239,68,68,0.14); border:1px solid rgba(239,68,68,0.35); }
      .footer { opacity:0.75; font-size: 12px; margin-top: 18px;}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Helpers
# =========================
def band(score: float) -> str:
    if score < 35:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"

def box_class(score: float) -> str:
    b = band(score)
    return "good" if b == "Low" else "warn" if b == "Medium" else "bad"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def train_pipeline(df: pd.DataFrame):
    # X & y
    X = df.drop(columns=["decision_id", "regret_index"])
    y = df["regret_index"]

    categorical_cols = ["experience_level", "decision_type", "abroad_intent"]
    numerical_cols = [
        "age", "years_experience", "urgency_level", "important_score", "complexity",
        "time_pressure", "effective_info_quality", "risk_aversion", "confidence_level",
        "numbers_of_options", "time_spent", "uncert_level"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=12,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # evaluation
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # version-safe

    return pipeline, {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}

def predict_regret(pipeline, input_df: pd.DataFrame) -> float:
    return float(pipeline.predict(input_df)[0])

def what_if(pipeline, base_df: pd.DataFrame, feature: str, new_value):
    modified = base_df.copy()
    modified[feature] = new_value
    base = predict_regret(pipeline, base_df)
    new = predict_regret(pipeline, modified)
    return base, new, (new - base)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =========================
# Header
# =========================
st.markdown(
    """
    <div class="hero">
      <h1>🔮 Regret Forecast Engine</h1>
      <p>Predict your regret risk <b>before</b> you decide — with clear insights + what-if simulation.</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Built by Moin Farid · Data Science & Machine Learning")

# =========================
# Load & train
# =========================
try:
    df = load_data("regret_dataset_1000.csv")
except Exception:
    st.error("CSV file not found. Put **regret_dataset_1000.csv** in the same folder as `app.py`.")
    st.stop()

pipeline, metrics = train_pipeline(df)

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("🧩 Your Inputs")

experience_level = st.sidebar.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced", "Expert"])
years_experience = st.sidebar.slider("Years of Experience", 0, 12, 3)

age = st.sidebar.slider("Age", 18, 55, 28)

decision_type = st.sidebar.selectbox(
    "Decision Type",
    ["career", "finance", "relationship", "health", "purchase", "education", "travel"]
)

abroad_intent = st.sidebar.selectbox("Planning Abroad?", ["Yes", "No"])

urgency_level = st.sidebar.slider("Urgency Level", 1, 10, 5)
important_score = st.sidebar.slider("Decision Importance", 1, 10, 6)
complexity = st.sidebar.slider("Decision Complexity", 1, 10, 6)
time_pressure = st.sidebar.slider("Time Pressure", 1, 10, 5)

effective_info_quality = st.sidebar.slider("Information Quality", 1, 10, 6)
confidence_level = st.sidebar.slider("Confidence Level", 1, 10, 6)
risk_aversion = st.sidebar.slider("Risk Aversion", 1, 10, 6)

numbers_of_options = st.sidebar.slider("Number of Options", 2, 15, 4)
time_spent = st.sidebar.slider("Time Spent Thinking (minutes)", 5, 240, 60)
uncert_level = st.sidebar.slider("Uncertainty Level", 1, 10, 5)

st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Run Forecast", use_container_width=True)

# =========================
# Build input DF
# =========================
input_df = pd.DataFrame([{
    "age": age,
    "experience_level": experience_level,
    "years_experience": years_experience,
    "urgency_level": urgency_level,
    "decision_type": decision_type,
    "abroad_intent": abroad_intent,
    "important_score": important_score,
    "complexity": complexity,
    "time_pressure": time_pressure,
    "effective_info_quality": effective_info_quality,
    "risk_aversion": risk_aversion,
    "confidence_level": confidence_level,
    "numbers_of_options": numbers_of_options,
    "time_spent": time_spent,
    "uncert_level": uncert_level
}])

# =========================
# Main Layout
# =========================
colA, colB = st.columns([1.25, 1])

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Your Forecast")

    if run:
        score = predict_regret(pipeline, input_df)
        st.metric("Predicted Regret Index", f"{score:.1f} / 100")
        st.markdown(f'<div class="{box_class(score)}"><b>Regret Risk:</b> {band(score)}</div>', unsafe_allow_html=True)

        st.markdown("### 🧠 Quick Insights")
        insights = []

        if urgency_level >= 8 or time_pressure >= 8:
            insights.append("High urgency/time pressure may push a rushed decision.")
        if effective_info_quality <= 4:
            insights.append("Information quality looks low — missing details often create regret later.")
        if confidence_level <= 4:
            insights.append("Low confidence increases post-decision doubt.")
        if uncert_level >= 7:
            insights.append("High uncertainty is a major regret driver.")
        if numbers_of_options >= 10:
            insights.append("Too many options can cause overthinking and choice regret.")
        if time_spent >= 150:
            insights.append("Overthinking can increase regret after the decision.")
        if time_spent <= 25 and complexity >= 7:
            insights.append("Too little thinking time for a complex decision can raise regret.")

        if not insights:
            insights.append("Inputs look balanced. Best improvement lever is usually info quality and uncertainty reduction.")

        for s in insights[:7]:
            st.write("•", s)

        st.markdown("### 🧪 What-If Simulation (How to reduce regret)")
        base = input_df.copy()

        suggestions = []

        # Reduce urgency
        _, _, d = what_if(pipeline, base, "urgency_level", clamp(urgency_level - 3, 1, 10))
        suggestions.append(("Lower urgency (−3)", d))

        # Improve info quality
        _, _, d = what_if(pipeline, base, "effective_info_quality", clamp(effective_info_quality + 2, 1, 10))
        suggestions.append(("Improve info quality (+2)", d))

        # Improve confidence
        _, _, d = what_if(pipeline, base, "confidence_level", clamp(confidence_level + 2, 1, 10))
        suggestions.append(("Increase confidence (+2)", d))

        # Reduce options
        _, _, d = what_if(pipeline, base, "numbers_of_options", clamp(numbers_of_options - 3, 2, 15))
        suggestions.append(("Reduce options (−3)", d))

        # Sort best improvements first (most negative delta)
        suggestions = sorted(suggestions, key=lambda x: x[1])

        for name, delta in suggestions[:4]:
            arrow = "↓" if delta < 0 else "↑"
            st.write(f"• {name}: predicted regret {arrow} {abs(delta):.1f}")

        st.markdown('<div class="small">This is decision-support, not professional advice.</div>', unsafe_allow_html=True)

    else:
        st.info("Set your inputs on the left and click **Run Forecast**.")
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Model Quality (Unseen Test Data)")
    st.write("Performance measured on **unseen** 20% test data:")

    st.write(f"• R²: **{metrics['r2']:.3f}**")
    st.write(f"• MAE: **{metrics['mae']:.2f}**")
    st.write(f"• RMSE: **{metrics['rmse']:.2f}**")

    st.markdown("### 🧾 Meaning (easy)")
    st.write("• **R²**: model kitna pattern explain karta hai (higher better).")
    st.write("• **MAE**: average prediction mistake (points).")
    st.write("• **RMSE**: badi mistakes ko zyada punish karta hai.")

    st.markdown("### ✅ What makes this project strong")
    st.write("• Dataset designed by behavior rules (not random numbers).")
    st.write("• Predictions + what-if guidance (not just a score).")
    st.write("• Explainable drivers (urgency, info, uncertainty, confidence).")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer">Regret Forecast Engine · v1</div>', unsafe_allow_html=True)
