# 🔮 Regret Forecast Engine

An **AI-powered decision support system** that predicts **post-decision regret** before you decide — and shows **how to reduce it** using what-if simulations.

This project goes beyond traditional prediction models by combining:
- Regression modeling
- Behavioral insights
- Scenario-based analysis

---

## 🚀 Live Demo
👉 *(Add your Streamlit app link here)*  
Example: https://your-app-name.streamlit.app

---

## 🧠 Why This Project?

Most decision tools answer:
> “What should I choose?”

This system answers a deeper question:
> **“How much regret will I feel later — and how can I reduce it now?”**

It models **human decision-making behavior** instead of just numbers.

---

## ⚙️ What the App Does

- Predicts a **Regret Index (0–100)** for a given decision
- Categorizes regret risk as **Low / Medium / High**
- Identifies **key regret drivers** (urgency, uncertainty, information quality, confidence)
- Runs **what-if simulations** to show how changing inputs can reduce regret
- Displays **model quality metrics** (R², MAE, RMSE) on unseen data

---

## 📊 Dataset

- **Type:** Synthetic, behavior-inspired dataset
- **Size:** 1000+ records
- **Design Philosophy:**  
  Based on real-world cognitive and situational decision patterns

### Key Features
- age  
- experience_level  
- years_experience  
- urgency_level  
- decision_type  
- abroad_intent  
- important_score  
- complexity  
- time_pressure  
- effective_info_quality  
- risk_aversion  
- confidence_level  
- numbers_of_options  
- time_spent  
- uncert_level  

**Target Variable:**  
- `regret_index` (continuous, 0–100)

---

## 🤖 Machine Learning Model

- **Model:** Random Forest Regressor  
- **Why:**  
  Captures non-linear relationships and interactions between human factors

### Evaluation (on unseen test data)
- **R²
