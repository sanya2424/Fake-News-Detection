"""Streamlit web application for Fake News Detection.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    FIGURES_DIR,
    METADATA_PATH,
    PROCESSED_DATA_PATH,
    REPORTS_DIR,
    SAMPLE_OUTPUTS_DIR,
)
from src.predict import append_prediction_history, load_prediction_artifacts, predict_news


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1180px;
    }
    .metric-card {
        border: 1px solid #d9dee8;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
    }
    .prediction-real {
        border-left: 6px solid #1f9d55;
        padding: 1rem;
        background: #f0fff6;
        border-radius: 8px;
    }
    .prediction-fake {
        border-left: 6px solid #d64545;
        padding: 1rem;
        background: #fff5f5;
        border-radius: 8px;
    }
    .small-muted {
        color: #5f6b7a;
        font-size: 0.9rem;
    }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def cached_artifacts() -> tuple[object, object, dict]:
    """Load model artifacts once per Streamlit session."""
    return load_prediction_artifacts()


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Load the processed dataset for the Dataset page."""
    if not PROCESSED_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_data
def load_model_results() -> pd.DataFrame:
    """Load model comparison results for the Model Results page."""
    results_path = REPORTS_DIR / "model_comparison.csv"
    if not results_path.exists():
        return pd.DataFrame()
    return pd.read_csv(results_path)


def show_header() -> None:
    """Display the shared app heading."""
    st.title("Fake News Detection using NLP and Machine Learning")
    st.caption(
        "Classify news articles as Fake or Real using TF-IDF features and "
        "classical machine learning models."
    )


def render_prediction_page() -> None:
    """Render the main article prediction interface."""
    show_header()

    try:
        model, vectorizer, metadata = cached_artifacts()
    except Exception as exc:
        st.error("Model artifacts are missing or could not be loaded.")
        st.info("Run `python -m src.run_pipeline` before starting the app.")
        st.exception(exc)
        return

    best_model_name = metadata.get("best_model_name", "Saved model")
    st.markdown(f"**Active model:** `{best_model_name}`")

    default_text = (
        "The election commission announced official results after all polling "
        "stations completed vote counting under independent observation."
    )
    article_text = st.text_area(
        "Paste or type a news article",
        value=default_text,
        height=260,
        placeholder="Enter the full news article text here...",
    )

    col_predict, col_clear = st.columns([1, 1])
    predict_clicked = col_predict.button("Predict", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("Clear History", use_container_width=True)

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if clear_clicked:
        st.session_state.prediction_history = []
        st.success("Prediction history cleared for this session.")

    if predict_clicked:
        try:
            with st.spinner("Cleaning text, vectorizing, and predicting..."):
                result = predict_news(article_text, model=model, vectorizer=vectorizer)
                st.session_state.prediction_history.append(result)
                append_prediction_history(
                    result,
                    SAMPLE_OUTPUTS_DIR / "prediction_history.csv",
                )

            prediction_class = (
                "prediction-real" if result["prediction"] == "Real" else "prediction-fake"
            )
            st.markdown(
                f"""
                <div class="{prediction_class}">
                    <h3>Prediction: {result["prediction"]}</h3>
                    <p><strong>Confidence:</strong> {result["confidence"] * 100:.2f}%</p>
                    <p>{result["explanation"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prob_col_1, prob_col_2 = st.columns(2)
            prob_col_1.metric("Fake Probability", f"{result['fake_probability'] * 100:.2f}%")
            prob_col_2.metric("Real Probability", f"{result['real_probability'] * 100:.2f}%")
            st.progress(result["confidence"])

            with st.expander("Show cleaned text used by the model"):
                st.write(result["cleaned_text"])

        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error("Something went wrong during prediction.")
            st.exception(exc)

    st.subheader("Prediction History")
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        display_columns = [
            "timestamp",
            "prediction",
            "confidence",
            "fake_probability",
            "real_probability",
            "input_text",
        ]
        st.dataframe(history_df[display_columns], use_container_width=True)
        st.download_button(
            "Download Prediction Results as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No predictions yet. Submit an article to build a session history.")


def render_dataset_page() -> None:
    """Render dataset preview and basic dataset statistics."""
    show_header()
    st.subheader("Dataset")

    df = load_processed_data()
    if df.empty:
        st.warning("Processed dataset not found. Run `python -m src.run_pipeline`.")
        return

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Articles", f"{len(df):,}")
    col_2.metric("Fake", f"{(df['label_name'] == 'Fake').sum():,}")
    col_3.metric("Real", f"{(df['label_name'] == 'Real').sum():,}")

    selected_label = st.selectbox("Filter by label", ["All", "Fake", "Real"])
    preview_df = df if selected_label == "All" else df[df["label_name"] == selected_label]

    columns_to_show = [
        column
        for column in ["title", "subject", "date", "label_name", "article_length", "clean_text"]
        if column in preview_df.columns
    ]
    st.dataframe(preview_df[columns_to_show].head(200), use_container_width=True)


def render_preprocessing_page() -> None:
    """Show the NLP preprocessing workflow."""
    show_header()
    st.subheader("Preprocessing Steps")
    steps = [
        "Combine title and article text into one input field.",
        "Handle missing values and remove duplicate articles.",
        "Convert text to lowercase.",
        "Remove URLs, HTML tags, emails, punctuation, numbers, and special characters.",
        "Tokenize text into words.",
        "Remove English stopwords.",
        "Lemmatize tokens to reduce words to their base form.",
        "Transform cleaned text using TF-IDF with unigram and bigram features.",
    ]
    for index, step in enumerate(steps, start=1):
        st.write(f"{index}. {step}")


def render_eda_page() -> None:
    """Display saved EDA visualizations."""
    show_header()
    st.subheader("Exploratory Data Analysis")

    figure_files = [
        "class_distribution.png",
        "article_length_distribution.png",
        "subject_distribution.png",
        "correlation_heatmap.png",
        "fake_news_wordcloud.png",
        "real_news_wordcloud.png",
        "top_words_fake.png",
        "top_words_real.png",
        "top_bigrams.png",
        "top_trigrams.png",
    ]

    existing_figures = [FIGURES_DIR / file_name for file_name in figure_files if (FIGURES_DIR / file_name).exists()]
    if not existing_figures:
        st.warning("EDA figures not found. Run `python -m src.run_pipeline`.")
        return

    for figure_path in existing_figures:
        st.image(str(figure_path), caption=figure_path.stem.replace("_", " ").title())


def render_results_page() -> None:
    """Display model comparison and evaluation reports."""
    show_header()
    st.subheader("Model Results")

    results_df = load_model_results()
    if results_df.empty:
        st.warning("Model comparison file not found. Run `python -m src.run_pipeline`.")
        return

    st.dataframe(results_df, use_container_width=True)

    comparison_plot = FIGURES_DIR / "model_comparison.png"
    best_matrix = FIGURES_DIR / "best_model_confusion_matrix.png"
    if comparison_plot.exists():
        st.image(str(comparison_plot), caption="Model Performance Comparison")
    if best_matrix.exists():
        st.image(str(best_matrix), caption="Best Model Confusion Matrix")

    report_path = REPORTS_DIR / "classification_report.txt"
    if report_path.exists():
        st.text("Classification Report")
        st.code(report_path.read_text(encoding="utf-8"), language="text")


def main() -> None:
    """Main Streamlit router."""
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            [
                "Predict",
                "Dataset",
                "Preprocessing",
                "EDA Visualizations",
                "Model Results",
            ],
        )
        st.markdown("---")
        st.caption("Built for a B.Tech AI/ML final-year NLP project.")

    if page == "Predict":
        render_prediction_page()
    elif page == "Dataset":
        render_dataset_page()
    elif page == "Preprocessing":
        render_preprocessing_page()
    elif page == "EDA Visualizations":
        render_eda_page()
    elif page == "Model Results":
        render_results_page()


if __name__ == "__main__":
    main()
