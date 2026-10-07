"""AI2 — failure-cause classifier.

Training and inference are separated:
- running this file directly trains and saves failure_type_model.joblib;
- importing this module does NOT read the CSV and does NOT retrain the model;
- predict_failure() loads the saved model lazily.
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

# RNF is random/noise in AI4I and is intentionally excluded from prediction.
TARGETS = ["TWF", "HDF", "PWF", "OSF"]
THRESHOLD = 0.30

MODEL_PATH = Path(os.getenv("FAILURE_TYPE_MODEL_PATH", Path(__file__).with_name("failure_type_model.joblib")))
DEFAULT_CSV_PATH = Path(os.getenv("AI4I_CSV_PATH", Path(__file__).with_name("ai4i2020.csv")))


def encode_type(type_: str) -> int:
    key = str(type_).strip().upper()
    if key not in TYPE_MAP:
        raise ValueError("type_ must be one of: L, M, H")
    return TYPE_MAP[key]


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["Overstrain_factor"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["Torque_to_Speed_ratio"] = df["Torque [Nm]"] / df["Rotational speed [rpm]"]
    return df


def prepare_training_data(csv_path: str | os.PathLike = DEFAULT_CSV_PATH):
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
    y = df[TARGETS]
    return df, X, y


def train_model(
    csv_path: str | os.PathLike = DEFAULT_CSV_PATH,
    model_path: str | os.PathLike = MODEL_PATH,
):
    """Train AI2 once, evaluate it and save failure_type_model.joblib."""
    df, X, y = prepare_training_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=df["Machine failure"],
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)
    predictions = pd.DataFrame(
        {
            target: (proba[:, 1] >= THRESHOLD).astype(int)
            if proba.shape[1] > 1
            else [0] * len(X_test)
            for target, proba in zip(TARGETS, proba_test)
        },
        index=X_test.index,
    )
    print(classification_report(y_test, predictions, target_names=TARGETS, zero_division=0))

    payload = {
        "model": model,
        "features": FEATURES,
        "targets": TARGETS,
        "threshold": THRESHOLD,
        "type_map": TYPE_MAP,
    }
    model_path = Path(model_path)
    joblib.dump(payload, model_path)
    print(f"Saved AI2 model to: {model_path}")

    load_model.cache_clear()
    return payload


@lru_cache(maxsize=1)
def load_model(model_path: str | os.PathLike = MODEL_PATH):
    """Load the saved AI2 model once per process."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"AI2 model not found: {path}. Run `python failurecause.py` first "
            "to create failure_type_model.joblib."
        )

    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload

    return {
        "model": payload,
        "features": FEATURES,
        "targets": TARGETS,
        "threshold": THRESHOLD,
        "type_map": TYPE_MAP,
    }


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


def predict_failure(type_, air_k, proc_k, rpm, torque, wear, threshold=THRESHOLD):
    """Return ({cause: probability}, [causes above threshold])."""
    row = _make_feature_row(type_, air_k, proc_k, rpm, torque, wear)
    payload = load_model()
    model = payload["model"]
    targets = payload.get("targets", TARGETS)

    proba_raw = model.predict_proba(row)
    proba = {}

    for target, output_proba, classes in zip(targets, proba_raw, model.classes_):
        classes = list(classes)
        if 1 in classes:
            p1 = float(output_proba[0][classes.index(1)])
        elif classes == [1]:
            p1 = 1.0
        else:
            p1 = 0.0
        proba[str(target)] = p1

    alarms = [target for target, p in proba.items() if p >= float(threshold)]
    return proba, alarms


if __name__ == "__main__":
    train_model()
