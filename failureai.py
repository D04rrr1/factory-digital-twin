"""AI1 — binary failure predictor.

Training and inference are separated:
- running this file directly trains the model and saves failure_binary_model.joblib;
- importing this module does NOT read the CSV and does NOT retrain the model;
- predict_failure_risk() loads the saved model lazily and returns P(failure).

Expected raw inputs:
    type_ : L | M | H
    air_k, proc_k, rpm, torque, wear
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_diff",
    "Power",
    "Overstrain_factor",
    "Torque_to_Speed_ratio",
]

MODEL_PATH = Path(os.getenv("FAILURE_BINARY_MODEL_PATH", Path(__file__).with_name("failure_binary_model.joblib")))
DEFAULT_CSV_PATH = Path(os.getenv("AI4I_CSV_PATH", Path(__file__).with_name("ai4i2020.csv")))


def encode_type(type_: str) -> int:
    """Convert machine type L/M/H to the numeric value used during training."""
    key = str(type_).strip().upper()
    if key not in TYPE_MAP:
        raise ValueError("type_ must be one of: L, M, H")
    return TYPE_MAP[key]


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features used by AI1 and AI3.

    The function returns a copy and does not mutate the caller's DataFrame.
    """
    df = frame.copy()
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["Overstrain_factor"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["Torque_to_Speed_ratio"] = df["Torque [Nm]"] / df["Rotational speed [rpm]"]
    return df


def prepare_training_data(csv_path: str | os.PathLike = DEFAULT_CSV_PATH):
    """Read AI4I 2020 data and return X, y ready for model training."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Put ai4i2020.csv next to this file "
            "or set AI4I_CSV_PATH."
        )

    df = pd.read_csv(csv_path)
    df = df.drop(columns=["UDI", "Product ID"])
    df["Type"] = df["Type"].map(TYPE_MAP)
    df = add_engineered_features(df)

    X = df[FEATURES]
    y = df["Machine failure"]
    return X, y


def train_model(
    csv_path: str | os.PathLike = DEFAULT_CSV_PATH,
    model_path: str | os.PathLike = MODEL_PATH,
):
    """Train AI1 once, evaluate it and save the fitted model to .joblib."""
    X, y = prepare_training_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, zero_division=0))

    payload = {
        "model": model,
        "features": FEATURES,
        "type_map": TYPE_MAP,
    }
    model_path = Path(model_path)
    joblib.dump(payload, model_path)
    print(f"Saved AI1 model to: {model_path}")

    # Refresh the lazy cache if training and inference happen in one process.
    load_model.cache_clear()
    return payload


@lru_cache(maxsize=1)
def load_model(model_path: str | os.PathLike = MODEL_PATH):
    """Load the saved AI1 model once per process."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"AI1 model not found: {path}. Run `python failureai.py` first "
            "to create failure_binary_model.joblib."
        )

    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload

    # Backward compatibility if somebody saved only the sklearn estimator.
    return {"model": payload, "features": FEATURES, "type_map": TYPE_MAP}


def _make_feature_row(type_, air_k, proc_k, rpm, torque, wear) -> pd.DataFrame:
    rpm = float(rpm)
    if rpm <= 0:
        raise ValueError("rpm must be greater than 0")

    row = pd.DataFrame([
        {
            "Type": encode_type(type_),
            "Air temperature [K]": float(air_k),
            "Process temperature [K]": float(proc_k),
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": float(torque),
            "Tool wear [min]": float(wear),
        }
    ])
    return add_engineered_features(row)[FEATURES]


def predict_failure_risk(type_, air_k, proc_k, rpm, torque, wear) -> float:
    """Return current failure probability as a float in the range 0..1."""
    X = _make_feature_row(type_, air_k, proc_k, rpm, torque, wear)
    model = load_model()["model"]
    proba = model.predict_proba(X)

    # Normal binary classifier: columns correspond to classes_ (usually [0, 1]).
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return float(proba[0][classes.index(1)])

    # A degenerate fitted model containing only class 0 or only class 1.
    if classes == [1]:
        return 1.0
    return 0.0


if __name__ == "__main__":
    train_model()
