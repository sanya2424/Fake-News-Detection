# Fake News Detection using NLP and Machine Learning

A complete machine learning project that classifies news articles as **Fake**
or **Real** using Natural Language Processing, TF-IDF vectorization, and
multiple supervised machine learning algorithms.

This project is designed for a 4th-year B.Tech AI/ML student and is structured
for college submission, GitHub portfolio use, internship demonstrations, and
technical interviews.

## Objectives

- Build an end-to-end fake news detection system.
- Clean and preprocess news text using standard NLP methods.
- Perform detailed Exploratory Data Analysis.
- Train and compare multiple machine learning models.
- Save the best model and TF-IDF vectorizer separately.
- Provide a responsive Streamlit app for real-time predictions.

## Features

- Automatic loading of `Fake.csv` and `True.csv`
- ZIP extraction support for `Fake.csv.zip` and `True.csv.zip`
- Missing value handling and duplicate removal
- Text normalization with regex, NLTK tokenization, stopword removal, and
  lemmatization
- EDA charts for class distribution, article length, subject distribution,
  word clouds, common words, n-grams, and correlations
- TF-IDF feature extraction using unigram and bigram features
- Model training for:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Random Forest
  - Support Vector Machine
  - Passive Aggressive Classifier
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, confusion
  matrix, classification report, and cross validation
- Automatic best model selection
- Streamlit web app with prediction history and CSV download
- Jupyter Notebook covering the complete workflow

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- WordCloud
- Regex

## Dataset

The project uses the Fake and Real News Dataset containing news titles, article
text, subjects, dates, and labels.

Expected files:

```text
data/raw/Fake.csv
data/raw/True.csv
```

ZIP files are also supported:

```text
data/raw/Fake.csv.zip
data/raw/True.csv.zip
```

If the CSV files are missing but the ZIP files exist, the project extracts them
automatically.

## Project Structure

```text
fake-news-detection/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   ├── Fake.csv
│   │   └── True.csv
│   └── processed/
│       └── cleaned_news.csv
├── docs/
│   ├── DATASET.md
│   └── PROJECT_REPORT.md
├── models/
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_metadata.json
├── notebooks/
│   └── Fake_News_Detection_Project.ipynb
├── reports/
│   ├── classification_report.txt
│   ├── model_comparison.csv
│   └── figures/
├── sample_outputs/
│   ├── sample_predictions.csv
│   └── prediction_result_mock.png
└── src/
    ├── config.py
    ├── data_loader.py
    ├── eda.py
    ├── predict.py
    ├── preprocessing.py
    ├── run_pipeline.py
    └── train.py
```

## Installation

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Complete Pipeline

Run this command once to preprocess data, generate EDA reports, train models,
save model artifacts, and create sample outputs.

```bash
python -m src.run_pipeline
```

## Run the Streamlit App

```bash
streamlit run app.py
```

The app includes pages for:

- Real-time prediction
- Dataset preview
- Preprocessing workflow
- EDA visualizations
- Model comparison results

## Results

After running the pipeline, model metrics are saved in:

```text
reports/model_comparison.csv
reports/classification_report.txt
reports/figures/model_comparison.png
reports/figures/best_model_confusion_matrix.png
```

The best model is selected using F1-score and accuracy.

## Screenshots

Add your final app screenshots here after running Streamlit:

```text
sample_outputs/prediction_result_mock.png
reports/figures/model_comparison.png
reports/figures/best_model_confusion_matrix.png
```

## Limitations

- The model learns from the writing patterns in the dataset and may not verify
  factual truth directly.
- New topics, satire, and highly edited articles may be harder to classify.
- Confidence values for margin-based models are approximate unless calibrated.

## Future Improvements

- Add transformer models such as BERT or RoBERTa.
- Add source credibility and URL-based features.
- Use model explainability tools such as LIME or SHAP.
- Add a database-backed prediction history.
- Deploy the Streamlit app on Streamlit Community Cloud or Hugging Face Spaces.

## Author Notes

This project follows modular programming and PEP 8 style. Functions are kept
clear and reusable so the workflow is easy to explain during project evaluation
or technical interviews.
