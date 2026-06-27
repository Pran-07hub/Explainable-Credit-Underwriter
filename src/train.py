import numpy as np
import pandas as pd
import joblib as jb
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def train_model():

    df = pd.read_csv("data/application_train.csv")

    X = df.drop("TARGET", axis=1)
    
    X = preprocess(X)

    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

    scale_pos_weight = (y==0).sum() / (y==1).sum()

    model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.02,
        subsample=0.65,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train,y_train)

    probs = model.predict_proba(X_test)[:, 1]

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    auc = roc_auc_score(y_test,probs)

    print(f"ROC AUC: {auc:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))

    jb.dump(model,"Models/xgb_model.pkl")

    desc_df = pd.read_csv("data/HomeCredit_columns_description.csv",encoding="latin1")

    feature_desc = dict(zip(desc_df["Row"],desc_df["Description"]))

    jb.dump(feature_desc,"Models/feature_desc.pkl")

    print("Artifacts saved.")

def preprocess(df):

    df["DAYS_EMPLOYED"] = (df["DAYS_EMPLOYED"].replace(365243, np.nan))

    cols_to_delete = []

    cols_to_delete.append('SK_ID_CURR')

    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'column': df.columns,'missing_%': missing_percent})

    drop_cols = missing_df[missing_df['missing_%']>50].index
    cols_to_delete.extend(drop_cols)
    
    for col in df.columns:
        top_freq = (df[col].value_counts(normalize=True,dropna=False).values[0])

        if top_freq > 0.95:
            cols_to_delete.append(col)    

    df.drop(columns=cols_to_delete, inplace=True, errors='ignore')

    jb.dump(df.columns.tolist(),"Models/feature_columns.pkl")

    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=np.number).columns

    imputation_values = {}

    for col in num_cols:
        imputation_values[col] = df[col].median()

    for col in cat_cols:
        imputation_values[col] = df[col].mode()[0]
    
    cat_count = df[cat_cols].nunique()
    
    for col in cat_cols:
        df[col] = df[col].fillna('Missing')
    
    ohe_cols = cat_count[cat_count < 5].index.tolist()
    le_cols= cat_count[cat_count >= 5].index.tolist()

    transform = ColumnTransformer(transformers=[
        ('transf1', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), le_cols),
        ('transf2', OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32), ohe_cols)
    ], remainder='passthrough')

    df = transform.fit_transform(df)

    jb.dump(imputation_values,'Models/imputation_values.pkl')

    jb.dump(cols_to_delete,'Models/deleted_columns.pkl')

    jb.dump(transform,'Models/encoder.pkl')

    return pd.DataFrame(df,columns=transform.get_feature_names_out())

train_model()
