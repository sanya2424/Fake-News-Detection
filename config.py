"""Central project configuration.

Keeping paths and model settings in one file makes the project easier to
understand and avoids repeating hard-coded strings across scripts.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NLTK_DATA_DIR = DATA_DIR / "nltk_data"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SAMPLE_OUTPUTS_DIR = PROJECT_ROOT / "sample_outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

FAKE_CSV_PATH = RAW_DATA_DIR / "Fake.csv"
TRUE_CSV_PATH = RAW_DATA_DIR / "True.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_news.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# A moderate feature count keeps training fast enough for student laptops while
# still capturing the vocabulary needed for strong fake-news classification.
MAX_TFIDF_FEATURES = 15000
NGRAM_RANGE = (1, 2)
CV_SAMPLE_SIZE = 15000

LABEL_TO_ID = {"Fake": 0, "Real": 1}
ID_TO_LABEL = {0: "Fake", 1: "Real"}


def create_project_directories() -> None:
    """Create all folders used by the project if they do not already exist."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        NLTK_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        SAMPLE_OUTPUTS_DIR,
        DOCS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
