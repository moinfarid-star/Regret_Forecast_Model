import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================
# CONFIG
# =========================
AUTHOR_NAME = "Moin Farid"
DATA_FILE = "regret_dataset.csv"   # <-- change to regret_dataset_1000.csv if needed


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
      .pill { display:inline-block; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(140,140,140,0.35); margin-right: 6px; margin-bottom: 6px; }
      .footer { opacity:0.75; font-size: 12px; margin-top: 18px;}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Helpers
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def band(score: float) -> str:
    if score < 35:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"

def band_style(score: float) -> str:
    b = band(score)
    return "good" if b == "Low" else "warn" if b == "Medium" else "bad"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def train_pipeline(df: pd.DataFrame):
    # features/target
    if "decision_id" not in df.columns or "regret_index" not in df.columns:
        raise ValueError("CSV must include decision_id and regret_index columns.")

    X = df.drop(columns=["decision_id", "regret_index"])
    y = df["regret_index"]

    categorical_cols = ["experience_level", "decision_type", "abroad_intent"]
    numerical_cols = [
        "age", "years_experience", "urgency_level", "important_score", "complexity",
        "time_pressure", "effective_info_quality", "risk_aversion", "confidence_level",
        "numbers_of_options", "time_spent", "uncert_level"
    ]

    # Basic safety checks
    missing = [c for c in (categorical_cols + numerical_cols) if c not in X.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=260,
        max_depth=14,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    # evaluation on unseen test data
    pred_test = pipeline.predict(X_test)
    r2 = r2_score(y_test, pred_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = math.sqrt(mean_squared_error(y_test, pred_test))  # version-safe

    meta = {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "pred_test": pred_test,
        "metrics": {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)},
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
    }
    return meta


def predict_regret(pipeline, input_df: pd.DataFrame) -> float:
    return float(pipeline.predict(input_df)[0])


def what_if(pipeline, base_df: pd.DataFrame, feature: str, new_value):
    modified = base_df.copy()
    modified[feature] = new_value
    base = predict_regret(pipeline, base_df)
    new = predict_regret(pipeline, modified)
    return base, new, (new - base)


def build_insights(input_row: dict):
    """
    Quick human insights from inputs (not model explanation).
    """
    insights_en = []
    insights_ur = []

    urgency = input_row["urgency_level"]
    pressure = input_row["time_pressure"]
    infoq = input_row["effective_info_quality"]
    conf = input_row["confidence_level"]
    uncert = input_row["uncert_level"]
    options = input_row["numbers_of_options"]
    t = input_row["time_spent"]
    cx = input_row["complexity"]

    if urgency >= 8 or pressure >= 8:
        insights_en.append("High urgency/time pressure can push rushed decisions.")
        insights_ur.append("Urgency/time pressure zyada ho to decision jaldi mein hota hai, regret barh sakta hai.")

    if infoq <= 4:
        insights_en.append("Information quality looks low — missing details often create regret later.")
        insights_ur.append("Info quality low hai — details miss hon to baad mein regret aata hai.")

    if conf <= 4:
        insights_en.append("Low confidence increases post-decision doubt.")
        insights_ur.append("Confidence low ho to baad mein doubt aur regret barh jata hai.")

    if uncert >= 7:
        insights_en.append("High uncertainty is a major driver of regret.")
        insights_ur.append("Uncertainty zyada ho to regret ka chance bohat barh jata hai.")

    if options >= 10:
        insights_en.append("Too many options can cause overthinking and choice regret.")
        insights_ur.append("Options bohat zyada hon to overthinking hoti hai aur regret barhta hai.")

    if t >= 150:
        insights_en.append("Very high time spent may indicate overthinking.")
        insights_ur.append("Time bohat zyada spend ho raha hai — overthinking ka signal ho sakta hai.")

    if t <= 25 and cx >= 7:
        insights_en.append("Too little thinking time for a complex decision can raise regret.")
        insights_ur.append("Complex decision ke liye time bohat kam hai — regret ka risk barh sakta hai.")

    if not insights_en:
        insights_en.append("Inputs look balanced. Best lever is usually improving info quality and reducing uncertainty.")
        insights_ur.append("Inputs balanced lag rahe hain. Best cheez: info improve karo aur uncertainty kam karo.")

    return insights_en, insights_ur


def build_pdf_report(payload: dict) -> bytes:
    """
    ReportLab PDF generator: returns PDF bytes for st.download_button
    """
    from io import BytesIO
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    x = 2.0 * cm
    y = h - 2.0 * cm

    # Title + Author
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Regret Forecast Engine — Report")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Generated by: {payload['author']} | {payload['timestamp']}")
    y -= 0.8 * cm

    # Section helper
    def section(title):
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, title)
        y -= 0.5 * cm
        c.setFont("Helvetica", 10)

    def kv(key, val):
        nonlocal y
        c.drawString(x, y, f"{key}: {val}")
        y -= 0.45 * cm

    # Inputs
    section("Inputs")
    for k, v in payload["inputs"].items():
        kv(k, v)

    y -= 0.2 * cm

    # Results
    section("Results")
    for k, v in payload["results"].items():
        kv(k, v)

    y -= 0.2 * cm

    # Model quality
    section("Model Quality (Unseen Test Data)")
    for k, v in payload["model_quality"].items():
        kv(k, v)

    y -= 0.2 * cm

    # Insights
    section("Insights (English)")
    for s in payload["insights_en"][:7]:
        c.drawString(x, y, f"• {s}")
        y -= 0.45 * cm

    y -= 0.2 * cm
    section("Insights (Roman Urdu)")
    for s in payload["insights_ur"][:7]:
        c.drawString(x, y, f"• {s}")
        y -= 0.45 * cm

    # Footer
    y = 1.6 * cm
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(x, y, "Disclaimer: This is a decision-support tool, not professional advice.")

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# =========================
# Header
# =========================
st.markdown(