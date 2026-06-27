import numpy as np
# Monkey-patch numpy.isnan to handle NumPy 2.x compatibility issues with scikit-learn's encoders on object/string columns
_orig_isnan = np.isnan
def _safe_isnan(x, *args, **kwargs):
    try:
        return _orig_isnan(x, *args, **kwargs)
    except TypeError:
        if isinstance(x, np.ndarray) and x.dtype == object:
            return np.array([isinstance(val, float) and _orig_isnan(val) for val in x])
        return np.zeros_like(x, dtype=bool)
np.isnan = _safe_isnan

import joblib as jb
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.memo_generator import generate_memo
from src.predict import risk_predict
from src.preprocess import preprocess_data
from src.explain import get_explanation, plot_waterfall


feature_desc = jb.load("Models/feature_desc.pkl")
col_map = jb.load("Models/column_mapping.pkl")

def final_chain(applicant_data):

    if isinstance(applicant_data, pd.DataFrame):
        applicant_df = applicant_data.copy()
    else:
        applicant_df = pd.DataFrame([applicant_data])

    applicant_df = preprocess_data(applicant_df)

    risk_score = risk_predict(applicant_df)

    explanations = get_explanation(applicant_df)

    risk_factor_info = []

    for feature in explanations["risk_factors"]:

        actual_feature = col_map.get(feature)
        risk_factor_info.append(
            {
                "feature": actual_feature,
                "description": feature_desc.get(actual_feature),
                "value": applicant_df.iloc[0][feature]
            }
        )

    protective_factor_info = []

    for feature in explanations["protective_factors"]:

        actual_feature = col_map.get(feature)
        protective_factor_info.append(
            {
                "feature": actual_feature,
                "description": feature_desc.get(actual_feature),
                "value": applicant_df.iloc[0][feature]
            }
        )

    memo = generate_memo(risk_score[0],risk_factor_info,protective_factor_info)

    waterfall_plot = plot_waterfall(applicant_df)

    return {
    "risk_score": risk_score,
    "memo": memo,
    "waterfall_plot": waterfall_plot
}

if __name__ == '__main__':
    dff = pd.read_csv("Data/application_test.csv")
    applicant_df = dff.iloc[2]
    result = final_chain(applicant_df)
    print(result["risk_score"])
    print(result["memo"])
    result["waterfall_plot"].savefig("waterfall.png")