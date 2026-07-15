"""Prediction utilities shared by scripts and the Streamlit app."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, ID_TO_LABEL, METADATA_PATH, VECTORIZER_PATH
from src.preprocessing import preprocess_for_prediction


def load_prediction_artifacts() -> tuple[object, object, dict]:
    """Load the trained model, TF-IDF vectorizer, and metadata."""
    if not BEST_MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run `python -m src.run_pipeline` first."
        )

    model = joblib.load(BEST_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, vectorizer, metadata


def _scores_to_probabilities(model: object, features) -> np.ndarray:
    """Return class probabilities or a calibrated-looking fallback distribution."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[0]

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(features)[0])
        real_probability = 1 / (1 + np.exp(-score))
        return np.array([1 - real_probability, real_probability])

    prediction = int(model.predict(features)[0])
    probabilities = np.array([0.0, 0.0])
    probabilities[prediction] = 1.0
    return probabilities


def explain_prediction(label: str, confidence: float) -> str:
    """Create a short, student-friendly explanation for a prediction."""
    confidence_percent = confidence * 100
    if label == "Fake":
        return (
            f"The article's word patterns are closer to fake-news examples in "
            f"the training dataset. Model confidence: {confidence_percent:.2f}%."
        )
    return (
        f"The article's word patterns are closer to real-news examples in the "
        f"training dataset. Model confidence: {confidence_percent:.2f}%."
    )


def predict_news(text: str, model: object | None = None, vectorizer: object | None = None) -> dict:
    """Predict whether a news article is Fake or Real."""
    if not text or len(text.strip()) < 25:
        raise ValueError("Please enter at least 25 characters of news content.")

    if model is None or vectorizer is None:
        model, vectorizer, _ = load_prediction_artifacts()

    cleaned_text = preprocess_for_prediction(text)
    if not cleaned_text:
        raise ValueError("The text became empty after preprocessing. Please add more words.")

    features = vectorizer.transform([cleaned_text])
    prediction_id = int(model.predict(features)[0])
    probabilities = _scores_to_probabilities(model, features)
    probabilities = probabilities / probabilities.sum()

    label = ID_TO_LABEL[prediction_id]
    confidence = float(probabilities[prediction_id])

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text": text,
        "cleaned_text": cleaned_text,
        "prediction": label,
        "confidence": confidence,
        "fake_probability": float(probabilities[0]),
        "real_probability": float(probabilities[1]),
        "explanation": explain_prediction(label, confidence),
    }


def append_prediction_history(result: dict, history_path: Path) -> pd.DataFrame:
    """Append one prediction result to a CSV history file."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([result])
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        history_df = pd.concat([history_df, new_row], ignore_index=True)
    else:
        history_df = new_row
    history_df.to_csv(history_path, index=False)
    return history_df
