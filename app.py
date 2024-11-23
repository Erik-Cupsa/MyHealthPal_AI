# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import string
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained Keras model
try:
    model = load_model('sentiment_model.h5')
    app.logger.info("Keras model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading Keras model: {e}")

# Load the tokenizer
try:
    tokenizer = joblib.load('tokenizer.joblib')
    app.logger.info("Tokenizer loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading tokenizer: {e}")

# Load the label encoder
try:
    label_encoder = joblib.load('label_encoder.joblib')
    app.logger.info("Label encoder loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading label encoder: {e}")

# Preprocessing function (same as used during training)
def preprocess_text(text):
    # Remove URLs, mentions, hashtags, punctuation, and digits
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    # Lowercase the text
    text = text.lower()
    return text.strip()

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        app.logger.warning("Request content type is not application/json")
        return jsonify({'error': 'Invalid content type. Expected application/json.'}), 400

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        app.logger.warning("No text provided in the request.")
        return jsonify({'error': 'No text provided.'}), 400

    app.logger.info(f"Received text: {text}")

    # Preprocess the input text
    clean_text = preprocess_text(text)
    app.logger.info(f"Cleaned text: {clean_text}")

    try:
        # Tokenize and pad the text
        sequences = tokenizer.texts_to_sequences([clean_text])
        padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')  # Ensure maxlen matches training

        # Make prediction
        prediction = model.predict(padded_sequences)
        predicted_class = prediction.argmax(axis=1)[0]
        sentiment = label_encoder.inverse_transform([predicted_class])[0]
        app.logger.info(f"Predicted sentiment: {sentiment}")
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Error during prediction.'}), 500

    # Return the result
    return jsonify({'sentiment': sentiment}), 200

if __name__ == '__main__':
    app.run(debug=True)
