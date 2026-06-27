import joblib
# pyrefly: ignore [missing-import]
import shap
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/xgb_model.pkl")

explainer = shap.TreeExplainer(model)

def get_explanation(customer_df):

    shap_values = explainer.shap_values(customer_df)[0]

    df = pd.DataFrame({
        "feature": customer_df.columns,
        "shap_value": shap_values
    })

    risk_factors = df[df["shap_value"] > 0].sort_values("shap_value",ascending=False).head(3)
    protective_factors = df[df["shap_value"] < 0].sort_values("shap_value",ascending=True).head(3)

    return {
        "risk_factors": risk_factors["feature"].tolist(),
        "protective_factors": protective_factors["feature"].tolist()
    }

def plot_waterfall(customer_df):

    shap_values = explainer(customer_df)

    plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False
    )

    plt.tight_layout()

    return plt.gcf()
