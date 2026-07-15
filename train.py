"""Model training and evaluation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    ID_TO_LABEL,
    CV_SAMPLE_SIZE,
    MAX_TFIDF_FEATURES,
    METADATA_PATH,
    MODELS_DIR,
    NGRAM_RANGE,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    VECTORIZER_PATH,
    create_project_directories,
)


@dataclass
class ModelResult:
    """Container for model metrics."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    cv_f1_mean: float
    cv_f1_std: float


def build_models() -> dict[str, object]:
    """Return all candidate machine learning models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=80,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
        "Support Vector Machine": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def _positive_scores(model: object, x_values) -> np.ndarray | None:
    """Return scores for ROC-AUC when the model exposes probabilities or margins."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_values)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_values)
    return None


def _save_confusion_matrix(y_true, y_pred, model_name: str, file_name: str) -> None:
    """Save a labelled confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
        yticklabels=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Save a comparison chart for major evaluation metrics."""
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    plot_df = results_df.melt(
        id_vars="model",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, x="score", y="model", hue="metric", palette="Set2")
    plt.xlim(0, 1)
    plt.title("Model Performance Comparison")
    plt.xlabel("Score")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close()


def train_and_evaluate(processed_df: pd.DataFrame) -> tuple[object, TfidfVectorizer, pd.DataFrame]:
    """Train all models, save metrics, and persist the best model/vectorizer."""
    create_project_directories()

    x_train, x_test, y_train, y_test = train_test_split(
        processed_df["clean_text"],
        processed_df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=processed_df["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_df=0.95,
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model_results: list[ModelResult] = []
    trained_models: dict[str, object] = {}

    for model_name, model in build_models().items():
        print(f"Training {model_name}...")
        model.fit(x_train_tfidf, y_train)
        y_pred = model.predict(x_test_tfidf)

        scores = _positive_scores(model, x_test_tfidf)
        roc_auc = roc_auc_score(y_test, scores) if scores is not None else None
        if x_train_tfidf.shape[0] > CV_SAMPLE_SIZE:
            rng = np.random.default_rng(RANDOM_STATE)
            cv_indices = rng.choice(
                x_train_tfidf.shape[0],
                size=CV_SAMPLE_SIZE,
                replace=False,
            )
            x_cv = x_train_tfidf[cv_indices]
            y_cv = y_train.iloc[cv_indices]
        else:
            x_cv = x_train_tfidf
            y_cv = y_train

        cv_scores = cross_val_score(
            model,
            x_cv,
            y_cv,
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )

        model_results.append(
            ModelResult(
                name=model_name,
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred),
                recall=recall_score(y_test, y_pred),
                f1_score=f1_score(y_test, y_pred),
                roc_auc=roc_auc,
                cv_f1_mean=float(cv_scores.mean()),
                cv_f1_std=float(cv_scores.std()),
            )
        )
        trained_models[model_name] = model
        _save_confusion_matrix(
            y_test,
            y_pred,
            model_name,
            f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
        )

    results_df = pd.DataFrame([result.__dict__ for result in model_results])
    results_df = results_df.rename(columns={"name": "model"}).sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    )
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    _plot_model_comparison(results_df)

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(x_test_tfidf)

    report_text = classification_report(
        y_test,
        best_predictions,
        target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
    )
    (REPORTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    report_dict = classification_report(
        y_test,
        best_predictions,
        target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
        output_dict=True,
    )
    (REPORTS_DIR / "classification_report.json").write_text(
        json.dumps(report_dict, indent=4),
        encoding="utf-8",
    )
    _save_confusion_matrix(
        y_test,
        best_predictions,
        best_model_name,
        "best_model_confusion_matrix.png",
    )

    metadata = {
        "best_model_name": best_model_name,
        "label_mapping": ID_TO_LABEL,
        "tfidf_max_features": MAX_TFIDF_FEATURES,
        "ngram_range": NGRAM_RANGE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "best_metrics": results_df.iloc[0].to_dict(),
    }

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    return best_model, vectorizer, results_df


def load_processed_dataset(path: Path) -> pd.DataFrame:
    """Load the cleaned dataset from disk."""
    return pd.read_csv(path)
