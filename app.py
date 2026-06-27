# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd

# pyrefly: ignore [missing-import]
from src.main_chain import final_chain

st.set_page_config(
    page_title="Explainable Credit Underwriter",
    layout="wide"
)

st.title("🏦 Explainable Credit Underwriter")

st.markdown(
    """
    Predict applicant default risk and generate an
    AI-powered underwriting memo with SHAP explanations.
    """
)

st.header("Applicant Information")

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=200000.0
)

credit = st.number_input(
    "Credit Amount",
    min_value=0.0,
    value=500000.0
)

annuity = st.number_input(
    "Loan Annuity",
    min_value=0.0,
    value=25000.0
)

days_birth = st.number_input(
    "Days Since Birth",
    value=-15000
)

days_employed = st.number_input(
    "Days Employed",
    value=-2000
)

if st.button("Evaluate Applicant"):

    customer_df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [income],
        "AMT_CREDIT": [credit],
        "AMT_ANNUITY": [annuity],
        "DAYS_BIRTH": [days_birth],
        "DAYS_EMPLOYED": [days_employed]
    })

    with st.spinner("Generating underwriting assessment..."):

        result = final_chain(customer_df)

    # Handle either tuple or dict return
    if isinstance(result, tuple):
        memo, waterfall_fig = result

        risk_score = None

    else:
        memo = result["memo"]
        waterfall_fig = result["waterfall_plot"]
        risk_score = result.get("risk_score")

    st.divider()

    if risk_score is not None:

        st.metric(
            "Predicted Default Risk",
            f"{risk_score.item():.1%}"
        )

    st.subheader("SHAP Waterfall Explanation")

    st.pyplot(
        waterfall_fig,
        clear_figure=False
    )

    st.divider()

    st.subheader("Credit Review Memo")

    st.markdown(memo)