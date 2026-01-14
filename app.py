import pathlib
import joblib
import numpy as np
import streamlit as st


BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def load_artifacts():
    """
    Load model artifacts for inference.
    If they are missing (e.g. fresh deployment), trigger training once.
    """
    model_path = MODELS_DIR / "best_premium_model.joblib"
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    features_path = MODELS_DIR / "feature_columns.joblib"

    if not model_path.exists() or not scaler_path.exists() or not features_path.exists():
        # Lazy training on first deploy – avoids errors when models are not committed
        from src import train_pipeline

        train_pipeline.main()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
    return model, scaler, feature_cols


def compute_risk_category(premium: float) -> str:
    if premium < 5000:
        return "Low"
    if premium < 15000:
        return "Medium"
    return "High"


def main():
    st.set_page_config(
        page_title="Health Insurance Premium Predictor",
        page_icon="💊",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Gradient background and card styling
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1f2937 50%, #111827 100%);
            color: #e5e7eb;
        }
        .main {
            background: transparent;
        }
        .card {
            background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(17,24,39,0.98));
            border-radius: 18px;
            padding: 22px 26px;
            box-shadow: 0 20px 45px rgba(0,0,0,0.45);
            border: 1px solid rgba(148,163,184,0.25);
        }
        .metric-card {
            background: radial-gradient(circle at top left, rgba(96,165,250,0.18), rgba(15,23,42,0.98));
            border-radius: 18px;
            padding: 18px 22px;
            border: 1px solid rgba(59,130,246,0.4);
        }
        .risk-low {
            background: linear-gradient(135deg, rgba(22,163,74,0.2), rgba(6,95,70,0.6));
            border-left: 4px solid #22c55e;
        }
        .risk-medium {
            background: linear-gradient(135deg, rgba(234,179,8,0.2), rgba(180,83,9,0.6));
            border-left: 4px solid #eab308;
        }
        .risk-high {
            background: linear-gradient(135deg, rgba(248,113,113,0.2), rgba(185,28,28,0.75));
            border-left: 4px solid #f87171;
        }
        .risk-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .risk-pill.low { background: rgba(22,163,74,0.12); color: #4ade80; }
        .risk-pill.medium { background: rgba(234,179,8,0.12); color: #facc15; }
        .risk-pill.high { background: rgba(248,113,113,0.12); color: #fca5a5; }
        .premium-value {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .sub-label {
            font-size: 0.8rem;
            color: #9ca3af;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="padding-bottom: 0.5rem;">
            <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.16em; color:#60a5fa; font-weight: 600;">
                HealthIQ Labs
            </div>
            <div style="display:flex; align-items:flex-end; gap:0.5rem; margin-top:0.2rem;">
                <h2 style="color:#e5e7eb; margin:0;">Health Insurance Premium Studio</h2>
            </div>
            <div style="color:#9ca3af; margin-top:0.35rem; max-width:640px; font-size:0.92rem;">
                Interactive underwriting-grade premium prediction engine. Tune lifestyle, vitals, and risk factors to see real-time pricing and risk segmentation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, scaler, feature_cols = load_artifacts()

    with st.sidebar:
        st.markdown("### 🧬 Profile & Lifestyle")
        age = st.slider("Age (years)", 18, 70, 35)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        smoker = st.radio("Smoker", ["No", "Yes"], horizontal=True)
        bmi = st.slider("BMI", 16.0, 45.0, 27.5, step=0.1)
        children = st.slider("Number of Dependents", 0, 5, 1)

        st.markdown("### 🏃🏻‍♀️ Habits")
        exercise_hours = st.slider("Exercise (hours / week)", 0.0, 15.0, 4.0, step=0.5)
        diet_quality = st.slider("Diet Quality (1–10)", 1, 10, 6)
        alcohol_units = st.slider("Alcohol Units / Week", 0.0, 30.0, 3.0, step=0.5)
        sleep_hours = st.slider("Sleep (hours / day)", 4.0, 9.5, 7.0, step=0.25)

        st.markdown("### ❤️ Health & Medical")
        chronic_conditions = st.slider("Chronic Conditions Count", 0, 6, 1)
        hospital_visits = st.slider("Hospital Visits (last year)", 0, 10, 1)
        medication_cost = st.slider("Medication Cost (₹ / month)", 0, 20000, 2000, step=500)
        bp_systolic = st.slider("Systolic BP (mmHg)", 95, 190, 125)
        cholesterol = st.slider("Cholesterol (mg/dL)", 130, 320, 200)
        stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)

        st.markdown("### 📊 Preventive & Family")
        annual_income = st.slider("Annual Income (Lakh ₹)", 2.0, 40.0, 12.0, step=0.5)
        # Use checkbox for broader Streamlit version compatibility
        annual_checkup = st.checkbox("Has Annual Health Checkup", value=True)
        family_history = st.slider("Family History Risk (0–10)", 0, 10, 3)
        dental_visits = st.slider("Dental Visits (last year)", 0, 5, 1)
        screenings = st.slider("Screenings Completed", 0, 10, 2)

    # Build feature vector based on training feature order
    feature_map = {
        "age_years": age,
        "is_male": 1 if sex.lower() == "male" else 0,
        "is_smoker": 1 if smoker.lower() == "yes" else 0,
        "bmi": float(bmi),
        "children_count": int(children),
        "annual_income_lakh": float(annual_income),
        "exercise_hours_per_week": float(exercise_hours),
        "diet_quality_score": float(diet_quality),
        "chronic_conditions_count": int(chronic_conditions),
        "hospital_visits_last_year": int(hospital_visits),
        "medication_cost_monthly": float(medication_cost),
        "stress_level_index": float(stress_level),
        "alcohol_units_per_week": float(alcohol_units),
        "sleep_hours_per_day": float(sleep_hours),
        "bp_systolic_mmHg": float(bp_systolic),
        "cholesterol_mg_dl": float(cholesterol),
        "has_annual_health_checkup": 1 if annual_checkup else 0,
        "family_history_score": float(family_history),
        "dental_visits_last_year": int(dental_visits),
        "screenings_completed": int(screenings),
    }

    x_vec = np.array([[feature_map[col] for col in feature_cols]], dtype=float)

    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        # tree-based or linear models were trained on unscaled X in this pipeline
        use_scaled = False
    else:
        use_scaled = True

    if use_scaled:
        x_vec_scaled = scaler.transform(x_vec)
        premium_pred = float(model.predict(x_vec_scaled)[0])
    else:
        premium_pred = float(model.predict(x_vec)[0])

    risk_cat = compute_risk_category(premium_pred)

    # Contribution-style explanation using normalized feature importance
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.array(model.coef_).ravel())
    else:
        importances = None

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1.2])
    with col1:
        st.markdown("##### Estimated Monthly Premium")
        st.markdown(
            f"<div class='premium-value'>₹ {premium_pred:,.0f}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sub-label'>Indicative underwriting premium based on current profile.</div>",
            unsafe_allow_html=True,
        )

    with col2:
        pill_class = {
            "Low": "low",
            "Medium": "medium",
            "High": "high",
        }[risk_cat]
        st.markdown("##### Risk Category")
        st.markdown(
            f"<div class='risk-pill {pill_class}'>{risk_cat} Risk</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        risk_class = {
            "Low": "risk-low",
            "Medium": "risk-medium",
            "High": "risk-high",
        }[risk_cat]
        st.markdown(
            f"<div class='card {risk_class}'>",
            unsafe_allow_html=True,
        )
        st.markdown("##### Underwriting Snapshot")
        st.write(
            "- **Age band**: {} yrs".format(age)
        )
        st.write(
            "- **Lifestyle**: Diet {}, {} hrs / week exercise, {} hrs / day sleep".format(
                diet_quality, exercise_hours, sleep_hours
            )
        )
        st.write(
            "- **Medical**: {} chronic conditions, {} hospital visits last year".format(
                chronic_conditions, hospital_visits
            )
        )
        st.write(
            "- **Vitals**: BP {} mmHg, Cholesterol {} mg/dL".format(
                bp_systolic, cholesterol
            )
        )
        st.write(
            "- **Protection**: {} screenings, {} dental visits, annual checkup: {}".format(
                screenings,
                dental_visits,
                "Yes" if annual_checkup else "No",
            )
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("##### Key Contributing Factors")

        if importances is not None:
            norm_imp = importances / (importances.sum() + 1e-8)
            top_idx = np.argsort(norm_imp)[-5:][::-1]
            for i in top_idx:
                fname = feature_cols[i]
                weight = norm_imp[i]
                label_map = {
                    "age_years": "Age",
                    "is_male": "Sex",
                    "is_smoker": "Smoking",
                    "bmi": "BMI",
                    "children_count": "Dependents",
                    "annual_income_lakh": "Income",
                    "exercise_hours_per_week": "Exercise",
                    "diet_quality_score": "Diet Quality",
                    "chronic_conditions_count": "Chronic Conditions",
                    "hospital_visits_last_year": "Hospital Visits",
                    "medication_cost_monthly": "Medication Spend",
                    "stress_level_index": "Stress Level",
                    "alcohol_units_per_week": "Alcohol Use",
                    "sleep_hours_per_day": "Sleep",
                    "bp_systolic_mmHg": "Blood Pressure",
                    "cholesterol_mg_dl": "Cholesterol",
                    "has_annual_health_checkup": "Annual Checkup",
                    "family_history_score": "Family History",
                    "dental_visits_last_year": "Dental Visits",
                    "screenings_completed": "Screenings",
                }
                display_name = label_map.get(fname, fname)
                bar_width = int(100 * float(weight))
                st.markdown(
                    f"""
                    <div style="margin-bottom:0.45rem;">
                        <div style="font-size:0.82rem; color:#e5e7eb;">{display_name}</div>
                        <div style="background:#111827; border-radius:999px; height:7px; overflow:hidden;">
                            <div style="width:{bar_width}%; height:7px; border-radius:999px;
                                        background:linear-gradient(90deg,#38bdf8,#6366f1);">
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write(
                "Model-level feature importances are not available for this estimator."
            )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


