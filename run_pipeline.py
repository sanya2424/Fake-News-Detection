"""Run the complete project pipeline from raw data to saved model artifacts."""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_PATH, SAMPLE_OUTPUTS_DIR, create_project_directories
from src.data_loader import load_raw_dataset
from src.eda import run_full_eda
from src.predict import predict_news
from src.preprocessing import preprocess_dataframe, save_processed_dataset
from src.train import train_and_evaluate


def main() -> None:
    """Execute data loading, preprocessing, EDA, training, and sample prediction."""
    create_project_directories()

    if PROCESSED_DATA_PATH.exists():
        print(f"Loading processed dataset from {PROCESSED_DATA_PATH}...")
        processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        print("Loading dataset...")
        raw_df = load_raw_dataset()

        print("Preprocessing text...")
        processed_df = preprocess_dataframe(raw_df)
        save_processed_dataset(processed_df)
        print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")

    print("Generating EDA reports...")
    run_full_eda(processed_df)

    print("Training and evaluating models...")
    best_model, vectorizer, results_df = train_and_evaluate(processed_df)
    print(results_df)

    print("Saving sample predictions...")
    sample_texts = []
    for label_name in ["Fake", "Real"]:
        sample_text = processed_df.loc[
            processed_df["label_name"] == label_name,
            "combined_text",
        ].iloc[0]
        sample_texts.append(sample_text[:1200])
    sample_results = [
        predict_news(text, model=best_model, vectorizer=vectorizer) for text in sample_texts
    ]
    pd.DataFrame(sample_results).to_csv(
        SAMPLE_OUTPUTS_DIR / "sample_predictions.csv",
        index=False,
    )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
