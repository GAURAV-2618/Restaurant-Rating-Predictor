"""Shared data preparation, model pipelines, evaluation, and artifact utilities.

Keep this module beside app.py and restaurant_predict.ipynb. Both entry points use
these functions so feature engineering and evaluation stay consistent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
TARGET_COL = "Aggregate rating"
NUMERIC_FEATURES = [
    "Average Cost for two", "Price range", "Votes", "Latitude", "Longitude",
    "Number of Cuisines",
]
CATEGORICAL_FEATURES = [
    "City", "Country Code", "Has Table booking", "Has Online delivery",
]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ARTIFACT_VERSION = 1

REQUIRED_COLUMNS = [
    "Restaurant ID", "Restaurant Name", "Country Code", "City", "Address",
    "Locality", "Locality Verbose", "Longitude", "Latitude", "Cuisines",
    "Average Cost for two", "Currency", "Has Table booking",
    "Has Online delivery", "Is delivering now", "Switch to order menu",
    "Price range", "Aggregate rating", "Rating color", "Rating text", "Votes",
]


def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> None:
    missing = [c for c in (required or REQUIRED_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))


def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw records and engineer features, identically for app and notebook.

    Unrated records are excluded. Zero coordinates and zero costs are treated as
    placeholders. Coordinate medians are calculated per (Country Code, City),
    with a country median and then a global median as fallbacks.
    """
    df = raw_df.copy()
    validate_columns(df, ["Country Code", "City", "Latitude", "Longitude",
                          "Cuisines", "Average Cost for two", TARGET_COL,
                          "Price range", "Votes", "Has Table booking",
                          "Has Online delivery"])
    df["Cuisines"] = df["Cuisines"].fillna("Unknown").astype(str)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)].copy()

    for col in ["Latitude", "Longitude", "Average Cost for two", "Price range", "Votes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    placeholder_coords = df["Latitude"].eq(0) | df["Longitude"].eq(0)
    df.loc[placeholder_coords, ["Latitude", "Longitude"]] = np.nan
    df.loc[df["Average Cost for two"].eq(0), "Average Cost for two"] = np.nan

    group_cols = ["Country Code", "City"]
    for col in ["Latitude", "Longitude"]:
        city_median = df.groupby(group_cols, dropna=False)[col].transform("median")
        country_median = df.groupby("Country Code", dropna=False)[col].transform("median")
        df[col] = df[col].fillna(city_median).fillna(country_median)
        global_median = df[col].median()
        if pd.notna(global_median):
            df[col] = df[col].fillna(global_median)

    df["Number of Cuisines"] = df["Cuisines"].apply(
        lambda value: len([part for part in str(value).split(", ") if part.strip()])
    )
    return df


def get_features_target(processed_df: pd.DataFrame):
    """Return model matrix and target with canonical column ordering/types."""
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in processed_df.columns]
    if missing:
        raise ValueError("Processed data is missing modeling columns: " + ", ".join(missing))
    X = processed_df[FEATURE_COLS].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = pd.to_numeric(processed_df[TARGET_COL], errors="coerce")
    valid = y.notna()
    return X.loc[valid].copy(), y.loc[valid].copy()


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")


def get_candidate_estimators() -> dict[str, Any]:
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, min_samples_leaf=5),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    try:
        from xgboost import XGBRegressor
        estimators["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="reg:squarederror",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    except ImportError:
        pass
    return estimators


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


def evaluate_and_fit(processed_df: pd.DataFrame) -> dict[str, Any]:
    """Shared protocol: 80/20 holdout; 5-fold CV on train only; refit on all data.

    The best model is selected solely by mean CV MAE. Test metrics are reported
    for all candidates but do not participate in model selection.
    """
    X, y = get_features_target(processed_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"mae": "neg_mean_absolute_error",
               "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    cv_rows, test_rows = [], []
    estimators = get_candidate_estimators()

    for name, estimator in estimators.items():
        pipeline = make_pipeline(estimator)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring,
                                n_jobs=1, return_train_score=False, error_score="raise")
        maes, rmses, r2s = -scores["test_mae"], -scores["test_rmse"], scores["test_r2"]
        cv_rows.append({"Model": name, "MAE mean": maes.mean(), "MAE std": maes.std(ddof=1),
                        "RMSE mean": rmses.mean(), "RMSE std": rmses.std(ddof=1),
                        "R2 mean": r2s.mean(), "R2 std": r2s.std(ddof=1)})
        holdout_model = make_pipeline(get_candidate_estimators()[name]).fit(X_train, y_train)
        pred = holdout_model.predict(X_test)
        test_rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                          "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                          "R2": r2_score(y_test, pred)})

    cv_df = pd.DataFrame(cv_rows).sort_values("MAE mean").reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sort_values("MAE").reset_index(drop=True)
    best_name = str(cv_df.iloc[0]["Model"])
    # Final production pipelines are trained on all rated rows after evaluation.
    fitted_models = {name: make_pipeline(estimator).fit(X, y)
                     for name, estimator in estimators.items()}
    city_coordinates = processed_df.groupby(["Country Code", "City"], dropna=False)[
        ["Latitude", "Longitude"]].median()
    return {
        "artifact_version": ARTIFACT_VERSION,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_splits": N_SPLITS,
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "models": fitted_models,
        "cv_results": cv_df,
        "test_results": test_df,
        "best_model_name": best_name,
        "city_coordinates": city_coordinates,
        "training_rows": len(X),
    }


def save_model_artifact(bundle: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    if bundle.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Model artifact version is incompatible; rerun the notebook training cell.")
    for key in ["models", "best_model_name", "cv_results", "test_results", "city_coordinates"]:
        if key not in bundle:
            raise ValueError(f"Model artifact is missing required key: {key}")
    return bundle
