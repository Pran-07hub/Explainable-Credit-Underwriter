import joblib as jb

def risk_predict(customer_df):

    model = jb.load("Models/xgb_model.pkl")

    return model.predict_proba(customer_df)[:, 1]