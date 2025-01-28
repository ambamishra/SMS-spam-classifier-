# Email/SMS Spam Classifier

This project is a machine learning-based web application for classifying messages as spam or not spam. The application uses a pre-trained model and a text processing pipeline to predict the spam likelihood of a given message.

## Features

- Preprocesses text messages using tokenization, stemming, and removal of stopwords.
- Classifies messages as "Spam" or "Not Spam" using a trained machine learning model.
- Simple and interactive interface built using Streamlit.

## File Structure

- `app.py`: Main Streamlit application file containing the user interface and prediction logic.
- `model.pkl`: Pre-trained machine learning model for spam classification.
- `vectorizer.pkl`: TF-IDF vectorizer used to preprocess text data.
- `sms-spam-detection.ipynb`: Jupyter notebook containing model training and evaluation code.

## Text Preprocessing

The preprocessing steps include:
- Converting text to lowercase.
- Tokenizing the text.
- Removing stopwords and punctuation.
- Stemming words to their root forms.

## Model Details

The application uses a machine learning model trained on a spam dataset. The `sms-spam-detection.ipynb` notebook provides the training process details, including data preprocessing, feature extraction, and model evaluation.

## Acknowledgments

This project was built using open-source libraries like Streamlit, NLTK, and scikit-learn.

