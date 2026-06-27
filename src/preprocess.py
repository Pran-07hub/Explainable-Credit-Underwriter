import numpy as np
import pandas as pd
import joblib as jb

def drop_columns(df):
    cols_drop = jb.load('Models/deleted_columns.pkl')
    df.drop(columns=cols_drop, inplace=True, errors='ignore')
    return df

def replace_special_values(df):
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    return df

def fill_mising(df):
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna('Missing')
    return df

def encode_features(df):
    transform = jb.load('Models/encoder.pkl')
    res = transform.transform(df)
    return pd.DataFrame(res, columns=transform.get_feature_names_out())

def align_features(df):
    feature_columns = jb.load('Models/feature_columns.pkl')
    imputation_values = jb.load("models/imputation_values.pkl")
    df = df.reindex(columns=feature_columns)
    df = df.fillna(imputation_values)
    return df

def coerce_categorical_dtypes(df):
    df = df.copy()
    transform = jb.load('Models/encoder.pkl')
    cat_cols = list(transform.transformers[0][2]) + list(transform.transformers[1][2])
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df

def preprocess_data(df):
    df = coerce_categorical_dtypes(df)
    df = drop_columns(df)
    df = replace_special_values(df)
    df = fill_mising(df)
    df = align_features(df)
    df = encode_features(df)
    return df