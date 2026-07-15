"""Exploratory Data Analysis utilities.

Each function saves visual output into reports/figures so the project contains
ready-to-submit graphs after the pipeline is executed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from src.config import FIGURES_DIR, REPORTS_DIR, create_project_directories


sns.set_theme(style="whitegrid")


def _save_plot(path: Path) -> None:
    """Save the current Matplotlib figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Plot the count of fake and real articles."""
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="label_name", hue="label_name", palette="Set2")
    ax.set_title("Class Distribution")
    ax.set_xlabel("News Class")
    ax.set_ylabel("Number of Articles")
    if ax.legend_ is not None:
        ax.legend_.remove()
    _save_plot(FIGURES_DIR / "class_distribution.png")


def plot_article_length_distribution(df: pd.DataFrame) -> None:
    """Plot article length distributions for both classes."""
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="article_length",
        hue="label_name",
        bins=60,
        kde=True,
        palette="Set1",
    )
    plt.xlim(0, df["article_length"].quantile(0.98))
    plt.title("Article Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Article Count")
    _save_plot(FIGURES_DIR / "article_length_distribution.png")


def plot_subject_distribution(df: pd.DataFrame) -> None:
    """Plot subject/category counts when the column exists."""
    if "subject" not in df.columns:
        return

    plt.figure(figsize=(11, 6))
    subject_counts = df["subject"].value_counts().reset_index()
    subject_counts.columns = ["subject", "count"]
    sns.barplot(data=subject_counts, y="subject", x="count", hue="subject", palette="viridis")
    plt.title("News Subject Distribution")
    plt.xlabel("Number of Articles")
    plt.ylabel("Subject")
    plt.legend([], [], frameon=False)
    _save_plot(FIGURES_DIR / "subject_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlations for available numeric columns."""
    numeric_df = df[["label", "article_length", "clean_length"]].copy()
    plt.figure(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    _save_plot(FIGURES_DIR / "correlation_heatmap.png")


def generate_wordclouds(df: pd.DataFrame) -> None:
    """Generate separate word clouds for fake and real news."""
    for label_name in ["Fake", "Real"]:
        text = " ".join(df.loc[df["label_name"] == label_name, "clean_text"])
        wordcloud = WordCloud(
            width=1200,
            height=700,
            background_color="white",
            colormap="Set2" if label_name == "Real" else "Reds",
            max_words=180,
        ).generate(text)

        plt.figure(figsize=(12, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{label_name} News WordCloud")
        _save_plot(FIGURES_DIR / f"{label_name.lower()}_news_wordcloud.png")


def get_common_words(df: pd.DataFrame, label_name: str, top_n: int = 25) -> pd.DataFrame:
    """Return the most frequent words for one class."""
    class_text = " ".join(df.loc[df["label_name"] == label_name, "clean_text"])
    words = class_text.split()
    common_words = Counter(words).most_common(top_n)
    return pd.DataFrame(common_words, columns=["word", "frequency"])


def plot_common_words(df: pd.DataFrame) -> None:
    """Plot the most common words for fake and real news."""
    for label_name in ["Fake", "Real"]:
        common_df = get_common_words(df, label_name)
        plt.figure(figsize=(10, 7))
        sns.barplot(
            data=common_df,
            y="word",
            x="frequency",
            hue="word",
            palette="mako",
            legend=False,
        )
        plt.title(f"Top Words in {label_name} News")
        plt.xlabel("Frequency")
        plt.ylabel("Word")
        _save_plot(FIGURES_DIR / f"top_words_{label_name.lower()}.png")
        common_df.to_csv(REPORTS_DIR / f"top_words_{label_name.lower()}.csv", index=False)


def plot_top_ngrams(df: pd.DataFrame, ngram_range: tuple[int, int], file_stem: str) -> None:
    """Create a bar chart of the most frequent n-grams."""
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=25)
    matrix = vectorizer.fit_transform(df["clean_text"])
    frequencies = matrix.sum(axis=0).A1
    ngram_df = pd.DataFrame(
        {
            "ngram": vectorizer.get_feature_names_out(),
            "frequency": frequencies,
        }
    ).sort_values("frequency", ascending=False)

    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=ngram_df,
        y="ngram",
        x="frequency",
        hue="ngram",
        palette="crest",
        legend=False,
    )
    plt.title(f"Top {file_stem.replace('_', ' ').title()}")
    plt.xlabel("Frequency")
    plt.ylabel("N-gram")
    _save_plot(FIGURES_DIR / f"top_{file_stem}.png")
    ngram_df.to_csv(REPORTS_DIR / f"top_{file_stem}.csv", index=False)


def run_full_eda(df: pd.DataFrame) -> None:
    """Run every EDA plot and tabular report."""
    create_project_directories()
    plot_class_distribution(df)
    plot_article_length_distribution(df)
    plot_subject_distribution(df)
    plot_correlation_heatmap(df)
    generate_wordclouds(df)
    plot_common_words(df)
    plot_top_ngrams(df, (2, 2), "bigrams")
    plot_top_ngrams(df, (3, 3), "trigrams")

    summary = {
        "total_articles": int(len(df)),
        "fake_articles": int((df["label_name"] == "Fake").sum()),
        "real_articles": int((df["label_name"] == "Real").sum()),
        "average_article_length": float(df["article_length"].mean()),
        "average_clean_length": float(df["clean_length"].mean()),
    }
    pd.Series(summary).to_csv(REPORTS_DIR / "eda_summary.csv", header=["value"])
