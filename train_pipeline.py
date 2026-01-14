import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "insurance.csv"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"


def ensure_directories() -> None:
    """Ensure that required directories exist."""
    for d in [MODELS_DIR, FIGURES_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def generate_synthetic_base_data(n_samples: int = 1338, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic version of the Medical Cost Personal Dataset
    with base columns: age, sex, bmi, children, smoker, charges.
    """
    rng = np.random.default_rng(random_state)

    # age: 18–64, skewed a bit toward middle age
    age = rng.integers(18, 65, size=n_samples)

    # sex: roughly balanced
    sex = rng.choice(["male", "female"], size=n_samples)

    # bmi: normal-ish distribution with realistic bounds
    bmi = np.clip(rng.normal(loc=30, scale=6, size=n_samples), 16, 50)

    # children: 0–5 with more weight on lower counts
    children_probs = [0.4, 0.25, 0.15, 0.1, 0.06, 0.04]
    children = rng.choice(np.arange(0, 6), size=n_samples, p=children_probs)

    # smoker: ~20% smokers
    smoker = rng.choice(["yes", "no"], size=n_samples, p=[0.2, 0.8])

    # region: optional, but can help mimic original dataset (not required later)
    regions = ["southwest", "southeast", "northwest", "northeast"]
    region = rng.choice(regions, size=n_samples)

    # Base medical charges formula inspired by public examples
    base_charge = 2000
    age_factor = (age - 18) * 120
    bmi_factor = (bmi - 25) * 200
    children_factor = children * 400
    smoker_factor = np.where(smoker == "yes", 15000, 0)
    noise = rng.normal(0, 3000, size=n_samples)

    charges = (
        base_charge
        + age_factor
        + np.maximum(bmi_factor, 0)
        + children_factor
        + smoker_factor
        + noise
    )
    charges = np.clip(charges, 1000, None)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi.round(2),
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": charges.round(2),
        }
    )
    return df


def save_base_data_if_missing() -> pd.DataFrame:
    """
    If data/insurance.csv exists, load and return it.
    Otherwise, generate a synthetic base dataset, save it, and return it.
    """
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return df

    df = generate_synthetic_base_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def engineer_features(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Create engineered features and target monthly_premium_inr."""
    rng = np.random.default_rng(random_state)

    df = df.copy()

    # Base features
    df["age_years"] = df["age"]
    df["is_male"] = (df["sex"].str.lower() == "male").astype(int)
    df["is_smoker"] = (df["smoker"].str.lower() == "yes").astype(int)
    df["bmi"] = df["bmi"].astype(float)
    df["children_count"] = df["children"].astype(int)

    # Synthetic socio-lifestyle and health features
    # annual_income_lakh: 2–40 Lakh with log-normal-like spread
    income = rng.lognormal(mean=2.5, sigma=0.5, size=len(df))
    income = np.clip(income, 2, 40)
    df["annual_income_lakh"] = income

    # exercise_hours_per_week: generally 0–10, smokers tend to exercise less
    base_exercise = rng.normal(4, 2, size=len(df))
    smoker_penalty = df["is_smoker"] * rng.uniform(0.5, 2.0, size=len(df))
    exercise = np.clip(base_exercise - smoker_penalty, 0, 12)
    df["exercise_hours_per_week"] = exercise.round(1)

    # diet_quality_score: 1–10, higher for non-smokers and lower BMI
    diet = 7 - (df["bmi"] - 25) / 5 - df["is_smoker"] * 1.5
    diet += rng.normal(0, 1, size=len(df))
    df["diet_quality_score"] = np.clip(diet, 1, 10).round(1)

    # chronic_conditions_count: Poisson-like with higher age and BMI
    lam = 0.3 + (df["age_years"] - 40).clip(lower=0) / 30 + (df["bmi"] - 30).clip(lower=0) / 15
    chronic = rng.poisson(lam=lam)
    df["chronic_conditions_count"] = np.clip(chronic, 0, 6)

    # hospital_visits_last_year: more with chronic conditions, smokers, high age
    lam_visits = 0.2 + df["chronic_conditions_count"] * 0.5 + df["is_smoker"] * 0.4
    visits = rng.poisson(lam=lam_visits)
    df["hospital_visits_last_year"] = np.clip(visits, 0, 10)

    # medication_cost_monthly (INR): depends on chronic conditions & age
    med_cost = 200 + df["chronic_conditions_count"] * rng.uniform(300, 700, len(df))
    med_cost += (df["age_years"] - 45).clip(lower=0) * 20
    med_cost += rng.normal(0, 300, len(df))
    df["medication_cost_monthly"] = np.clip(med_cost, 100, 20000).round(0)

    # stress_level_index: 1–10, higher for smokers, more income (work), less sleep
    base_stress = 4 + df["is_smoker"] * 2 + (df["annual_income_lakh"] - 10) / 5
    base_stress += rng.normal(0, 1, len(df))
    # We'll correct with sleep later after we generate it; for now keep base

    # alcohol_units_per_week: 0–30, higher for smokers
    alcohol = rng.normal(4, 4, len(df)) + df["is_smoker"] * rng.uniform(2, 6, len(df))
    df["alcohol_units_per_week"] = np.clip(alcohol, 0, 35).round(1)

    # sleep_hours_per_day: 4–9, lower for stressed people
    sleep = rng.normal(7, 1, len(df)) - df["is_smoker"] * 0.3
    sleep -= (df["annual_income_lakh"] - 12) / 20
    df["sleep_hours_per_day"] = np.clip(sleep, 4, 9).round(1)

    # Recompute stress including sleep
    stress = base_stress - (df["sleep_hours_per_day"] - 7)
    df["stress_level_index"] = np.clip(stress, 1, 10).round(1)

    # bp_systolic_mmHg: 100–180, higher with age, bmi, smoker
    bp = 110 + (df["age_years"] - 30) * 0.5 + (df["bmi"] - 25) * 1.2 + df["is_smoker"] * 8
    bp += rng.normal(0, 8, len(df))
    df["bp_systolic_mmHg"] = np.clip(bp, 95, 190).round(0)

    # cholesterol_mg_dl: 150–300, higher with bmi, smoker, low exercise
    chol = 170 + (df["bmi"] - 25) * 3 + df["is_smoker"] * 15 - df["exercise_hours_per_week"] * 1.2
    chol += rng.normal(0, 15, len(df))
    df["cholesterol_mg_dl"] = np.clip(chol, 130, 320).round(0)

    # has_annual_health_checkup: binary, more likely for higher income & chronic
    prob_checkup = 0.3 + (df["annual_income_lakh"] - 8) / 40 + df["chronic_conditions_count"] * 0.05
    prob_checkup = np.clip(prob_checkup, 0.1, 0.95)
    df["has_annual_health_checkup"] = (rng.random(len(df)) < prob_checkup).astype(int)

    # family_history_score: 0–10, correlated with chronic_conditions_count
    fam = df["chronic_conditions_count"] * 1.5 + rng.normal(2, 1.5, len(df))
    df["family_history_score"] = np.clip(fam, 0, 10).round(1)

    # dental_visits_last_year: 0–4, more for higher income / checkups
    lam_dental = 0.3 + df["annual_income_lakh"] / 30 + df["has_annual_health_checkup"] * 0.5
    dental = rng.poisson(lam=np.clip(lam_dental, 0.1, 3.0))
    df["dental_visits_last_year"] = np.clip(dental, 0, 5)

    # screenings_completed: 0–8, more for older, higher income & checkups
    lam_screen = 0.5 + (df["age_years"] - 35).clip(lower=0) / 15
    lam_screen += df["annual_income_lakh"] / 25 + df["has_annual_health_checkup"] * 1.0
    screenings = rng.poisson(lam=np.clip(lam_screen, 0.2, 6.0))
    df["screenings_completed"] = np.clip(screenings, 0, 10)

    # Target: monthly_premium_inr derived from charges & risk profile
    # Base from annual charges => convert to monthly, then adjust
    base_premium = df["charges"] / 12

    risk_score = (
        0.03 * (df["age_years"] - 30).clip(lower=0)
        + 0.06 * (df["bmi"] - 25).clip(lower=0)
        + 0.25 * df["is_smoker"]
        + 0.06 * df["chronic_conditions_count"]
        + 0.03 * df["hospital_visits_last_year"]
        + 0.02 * (df["bp_systolic_mmHg"] - 120).clip(lower=0) / 10
        + 0.02 * (df["cholesterol_mg_dl"] - 200).clip(lower=0) / 20
        + 0.02 * (df["stress_level_index"] - 5).clip(lower=0)
        - 0.015 * (df["exercise_hours_per_week"] - 3).clip(lower=0)
        - 0.015 * (df["diet_quality_score"] - 5).clip(lower=0)
    )

    risk_multiplier = 1 + risk_score
    income_discount = np.clip(1.1 - df["annual_income_lakh"] / 60, 0.7, 1.1)

    monthly_premium = base_premium * risk_multiplier * income_discount
    monthly_premium += df["medication_cost_monthly"] * 0.1
    monthly_premium += rng.normal(0, 300, len(df))
    df["monthly_premium_inr"] = np.clip(monthly_premium, 500, 50000).round(0)

    return df


def get_feature_target(df: pd.DataFrame):
    """Split engineered dataframe into X, y."""
    feature_cols = [
        "age_years",
        "is_male",
        "is_smoker",
        "bmi",
        "children_count",
        "annual_income_lakh",
        "exercise_hours_per_week",
        "diet_quality_score",
        "chronic_conditions_count",
        "hospital_visits_last_year",
        "medication_cost_monthly",
        "stress_level_index",
        "alcohol_units_per_week",
        "sleep_hours_per_day",
        "bp_systolic_mmHg",
        "cholesterol_mg_dl",
        "has_annual_health_checkup",
        "family_history_score",
        "dental_visits_last_year",
        "screenings_completed",
    ]
    X = df[feature_cols].values
    y = df["monthly_premium_inr"].values
    return X, y, feature_cols


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all specified models and evaluate them."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LinearRegression": (LinearRegression(), X_train_scaled, X_test_scaled),
        "Ridge": (Ridge(alpha=1.0), X_train_scaled, X_test_scaled),
        "Lasso": (Lasso(alpha=0.001), X_train_scaled, X_test_scaled),
        "SVR": (SVR(kernel="rbf", C=10, epsilon=0.2), X_train_scaled, X_test_scaled),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
            ),
            X_train,
            X_test,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            ),
            X_train,
            X_test,
        ),
    }

    results = {}
    best_model_name = None
    best_r2 = -np.inf

    for name, (model, Xtr, Xte) in models.items():
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5


        results[name] = {"r2": r2, "mae": mae, "rmse": rmse, "model": model}

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    return results, best_model_name, scaler


def save_metrics(results: dict, best_model_name: str) -> None:
    """Save model metrics to a CSV file."""
    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "model": name,
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "is_best": name == best_model_name,
            }
        )
    df_metrics = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(OUTPUTS_DIR / "model_metrics.csv", index=False)


def plot_predicted_vs_actual(y_true, y_pred, model_name: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    plt.xlabel("Actual Monthly Premium (INR)")
    plt.ylabel("Predicted Monthly Premium (INR)")
    plt.title(f"Predicted vs Actual - {model_name}")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names) -> None:
    """Plot feature importances for tree-based or XGBoost models, or coefficients for linear."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importances = None
    title = "Feature Importance"

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importance"
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_).ravel()
        importances = np.abs(coef)
        title = "Absolute Coefficients Importance"

    if importances is None:
        return

    idx = np.argsort(importances)[::-1]
    sorted_importances = importances[idx]
    sorted_features = np.array(feature_names)[idx]

    plt.figure(figsize=(9, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()


def plot_pca_scree(X) -> None:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    exp_var = pca.explained_variance_ratio_
    components = np.arange(1, len(exp_var) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(components, np.cumsum(exp_var), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("PCA Scree Plot")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "pca_scree_plot.png", dpi=150)
    plt.close()


def main():
    ensure_directories()

    # Load or generate base data
    base_df = save_base_data_if_missing()

    # Engineer features & target
    full_df = engineer_features(base_df)

    # Persist engineered dataset
    full_df.to_csv(OUTPUTS_DIR / "engineered_dataset.csv", index=False)

    # Prepare train/test
    X, y, feature_cols = get_feature_target(full_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    results, best_model_name, scaler = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    # Save metrics
    save_metrics(results, best_model_name)

    # Best model & plots
    best_model = results[best_model_name]["model"]
    # Use scaled features for models that were trained on scaled data
    if best_model_name in ["LinearRegression", "Ridge", "Lasso", "SVR"]:
        X_test_scaled = scaler.transform(X_test)
        y_pred_best = best_model.predict(X_test_scaled)
    else:
        y_pred_best = best_model.predict(X_test)

    plot_predicted_vs_actual(y_test, y_pred_best, best_model_name)
    plot_feature_importance(best_model, feature_cols)
    plot_pca_scree(X)

    # Save best model and scaler
    joblib.dump(best_model, MODELS_DIR / "best_premium_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")

    # Save feature columns for app
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")

    print(f"Training complete. Best model: {best_model_name}")


if __name__ == "__main__":
    main()


