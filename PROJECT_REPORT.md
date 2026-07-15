# Fake News Detection using NLP and Machine Learning

## Objective

The objective is to classify a news article as Fake or Real using Natural
Language Processing and supervised machine learning.

## Workflow

1. Load fake and real news CSV files.
2. Add labels and combine both classes into one dataset.
3. Clean text using lowercasing, URL removal, HTML removal, punctuation removal,
   stopword removal, tokenization, and lemmatization.
4. Perform Exploratory Data Analysis.
5. Convert cleaned text to numerical features using TF-IDF.
6. Train and compare Logistic Regression, Multinomial Naive Bayes, Random
   Forest, Support Vector Machine, and Passive Aggressive Classifier.
7. Save the best model and TF-IDF vectorizer with Joblib.
8. Deploy the prediction workflow in a Streamlit web application.

## Evaluation Metrics

The project compares all models using accuracy, precision, recall, F1-score,
ROC-AUC where available, confusion matrices, and 3-fold cross validation.

## Deliverables

- Trained model: `models/best_model.joblib`
- TF-IDF vectorizer: `models/tfidf_vectorizer.joblib`
- Model metadata: `models/model_metadata.json`
- EDA graphs: `reports/figures/`
- Model comparison: `reports/model_comparison.csv`
- Classification report: `reports/classification_report.txt`
- Streamlit app: `app.py`
- Notebook: `notebooks/Fake_News_Detection_Project.ipynb`
