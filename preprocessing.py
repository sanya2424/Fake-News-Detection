"""Text preprocessing functions used by training and the Streamlit app."""

from __future__ import annotations

import re
import string
from typing import Iterable

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import NLTK_DATA_DIR, PROCESSED_DATA_PATH, create_project_directories


NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def download_nltk_resources() -> None:
    """Download NLTK resources quietly if they are not already installed."""
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))

    for package_name in NLTK_PACKAGES:
        try:
            nltk.download(
                package_name,
                download_dir=str(NLTK_DATA_DIR),
                quiet=True,
                raise_on_error=False,
            )
        except Exception:
            # The cleaning pipeline has a safe fallback tokenizer. If a machine
            # has no internet during a rerun, training can still proceed.
            continue


def _wordnet_available() -> bool:
    """Return True when the WordNet corpus can be loaded by NLTK."""
    try:
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        try:
            nltk.data.find("corpora/wordnet.zip")
            return True
        except LookupError:
            return False


def _safe_tokenize(text: str) -> list[str]:
    """Tokenize cleaned text.

    The text has already been normalized with regex, so whitespace tokenization
    is both correct and much faster for this large dataset. NLTK remains part of
    the pipeline through stopwords and WordNet lemmatization.
    """
    return text.split()


def clean_text(text: str, stop_words: set[str] | None = None) -> str:
    """Clean one news article using standard NLP preprocessing steps.

    Steps included:
    - lowercase conversion
    - URL and HTML tag removal
    - punctuation, number, and special-character removal
    - tokenization
    - stopword removal
    - lemmatization
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if stop_words is None:
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            stop_words = set()

    lemmatizer = WordNetLemmatizer()
    can_lemmatize = _wordnet_available()
    tokens = _safe_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token) if can_lemmatize else token
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(cleaned_tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the raw dataset for ML training and EDA."""
    download_nltk_resources()

    processed_df = df.copy()
    processed_df["title"] = processed_df["title"].fillna("")
    processed_df["text"] = processed_df["text"].fillna("")
    processed_df["combined_text"] = (
        processed_df["title"].astype(str) + " " + processed_df["text"].astype(str)
    )

    before_duplicates = len(processed_df)
    processed_df = processed_df.drop_duplicates(subset=["combined_text"])
    after_duplicates = len(processed_df)

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    processed_df["clean_text"] = processed_df["combined_text"].apply(
        lambda value: clean_text(value, stop_words=stop_words)
    )
    processed_df["article_length"] = processed_df["combined_text"].str.split().str.len()
    processed_df["clean_length"] = processed_df["clean_text"].str.split().str.len()

    processed_df = processed_df[processed_df["clean_text"].str.len() > 0].copy()
    processed_df.attrs["duplicates_removed"] = before_duplicates - after_duplicates
    return processed_df.reset_index(drop=True)


def save_processed_dataset(df: pd.DataFrame) -> None:
    """Save the cleaned dataset for reproducibility and app display."""
    create_project_directories()
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def preprocess_for_prediction(text: str) -> str:
    """Apply exactly the same cleaning logic to new user input."""
    download_nltk_resources()
    return clean_text(text)


def join_tokens(tokens: Iterable[str]) -> str:
    """Small helper used in notebooks and reports."""
    return " ".join(tokens)
