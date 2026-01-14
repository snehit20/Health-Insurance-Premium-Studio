## Health Insurance Premium Prediction System

This project is an end-to-end **Health Insurance Premium Prediction** system built with **Python**, **machine learning**, and a **Streamlit** web application.

### Project Structure

- `data/` – Input dataset (`insurance.csv`)
- `src/` – Data generation, feature engineering, and model training code
- `models/` – Saved trained model files (`.joblib`)
- `outputs/` – Evaluation metrics and logs
- `figures/` – Generated plots (Predicted vs Actual, feature importance, PCA scree)
- `app/` – Streamlit web application

### Quick Start

1. (Optional) Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run data generation + training pipeline (creates synthetic data if not present):

```bash
python -m src.train_pipeline
```

3. Launch the Streamlit app:

```bash
streamlit run app/app.py
```

The training script will:

- Generate a realistic synthetic version of the **Medical Cost Personal** dataset if `data/insurance.csv` is missing.
- Perform feature engineering and create a target `monthly_premium_inr`.
- Train multiple regression models, evaluate them, and select the best one.
- Save the best model to `models/best_premium_model.joblib`.
- Produce visualizations in `figures/`.


