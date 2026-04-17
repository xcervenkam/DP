from typing import Dict, Tuple
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_numeric_features(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
        ],
        remainder="drop",
    )


def get_models(numeric_features: list[str]) -> Dict[str, Pipeline]:
    preprocessor = build_preprocessor(numeric_features)

    models = {
        "logreg": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=2000, multi_class="multinomial"))
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", KNeighborsClassifier(n_neighbors=15))
            ]
        ),
        "naive_bayes": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", GaussianNB())
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", SVC(probability=True, kernel="rbf", C=1.0))
            ]
        ),
        "tree": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
            ]
        ),
        "rf": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42))
            ]
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="mlogloss",
                        random_state=42,
                    ),
                ),
            ]
        )

    return models


def fit_and_predict(models: Dict[str, Pipeline], X_train, y_train, X_test) -> Tuple[dict, dict]:
    fitted_models = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        predictions[name] = {
            "pred": model.predict(X_test),
            "proba": model.predict_proba(X_test),
        }

    return fitted_models, predictions