
# fairness_quickstart.py
# Minimaler, robuster Weg zu y_true, y_prob, y_pred, group – falls du kein `model` definiert hast.
# Passt sich automatisch an typische Tabellenspalten an.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def build_and_predict(df: pd.DataFrame, group_col: str = "gender", threshold: float = 0.5):
    # Zielvariable
    y = df["Hire_binary"].astype(int).to_numpy()
    group = df[group_col].astype(str).to_numpy()

    # Feature-Kandidaten (passe bei Bedarf an)
    possible_numeric = [
        "AI Score (0-100)", "Experience (Years)", "Projects Count", "Salary Expectation ($)",
        "num_skills"
    ]
    possible_categ = [
        "Education", "Job Role", "Recruiter Decision"
    ]

    numeric_features = [c for c in possible_numeric if c in df.columns]
    categorical_features = [c for c in possible_categ if c in df.columns]

    # Fallback: alles, was numerisch ist und nicht y/group, dazu nehmen
    if not numeric_features:
        numeric_features = [c for c in df.select_dtypes(include=["number"]).columns
                            if c not in ["Hire_binary"]]

    X = df[numeric_features + categorical_features].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop"
    )

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200, n_jobs=None))
    ])

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, group, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = dict(
        accuracy=accuracy_score(y_test, y_pred),
        auc=roc_auc_score(y_test, y_prob),
        f1=f1_score(y_test, y_pred)
    )

    return dict(
        X_test=X_test, y_true=y_test, y_prob=y_prob, y_pred=y_pred, group=g_test,
        model=model, metrics=metrics, feature_cols=numeric_features + categorical_features
    )
