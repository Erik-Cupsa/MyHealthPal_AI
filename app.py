from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import string
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr
import os

app = Flask(__name__)

# Enable CORS for all endpoints and origins
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained Keras model
try:
    print("Loading Keras model...")
    model = load_model('sentiment_model.h5')
    app.logger.info("Keras model loaded successfully.")
    print("Keras model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading Keras model: {e}")
    print(f"Error loading Keras model: {e}")

# Load the tokenizer
try:
    print("Loading tokenizer...")
    tokenizer = joblib.load('tokenizer.joblib')
    app.logger.info("Tokenizer loaded successfully.")
    print("Tokenizer loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading tokenizer: {e}")
    print(f"Error loading tokenizer: {e}")

# Load the label encoder
try:
    print("Loading label encoder...")
    label_encoder = joblib.load('label_encoder.joblib')
    app.logger.info("Label encoder loaded successfully.")
    print("Label encoder loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading label encoder: {e}")
    print(f"Error loading label encoder: {e}")

# Preprocessing function (same as used during training)
def preprocess_text(text):
    print("Preprocessing text...")
    # Remove URLs, mentions, hashtags, punctuation, and digits
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    # Lowercase the text
    text = text.lower()
    print(f"Preprocessed text: {text}")
    return text.strip()

@app.route('/predict', methods=['POST'])
def predict():
    print("Received /predict request.")
    if not request.is_json:
        app.logger.warning("Request content type is not application/json")
        print("Invalid content type. Expected application/json.")
        return jsonify({'error': 'Invalid content type. Expected application/json.'}), 400

    data = request.get_json()
    print(f"Request JSON data: {data}")
    text = data.get('text', '')

    if not text:
        app.logger.warning("No text provided in the request.")
        print("No text provided in the request.")
        return jsonify({'error': 'No text provided.'}), 400

    app.logger.info(f"Received text: {text}")
    print(f"Received text: {text}")

    # Preprocess the input text
    clean_text = preprocess_text(text)
    app.logger.info(f"Cleaned text: {clean_text}")
    print(f"Cleaned text: {clean_text}")

    try:
        # Tokenize and pad the text
        print("Tokenizing and padding text...")
        sequences = tokenizer.texts_to_sequences([clean_text])
        print(f"Tokenized sequences: {sequences}")
        padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
        print(f"Padded sequences: {padded_sequences}")

        # Make prediction
        print("Making prediction...")
        prediction = model.predict(padded_sequences)
        print(f"Raw prediction: {prediction}")

        # Determine confidence and sentiment
        predicted_class = prediction.argmax(axis=1)[0]
        confidence = float(prediction[0][predicted_class])  # Convert to Python float
        print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

        # Confidence threshold for neutrality
        confidence_threshold = 0.65  # Adjust based on testing
        if confidence < confidence_threshold:
            sentiment = "neutral"
        else:
            sentiment = label_encoder.inverse_transform([predicted_class])[0]

        app.logger.info(f"Predicted sentiment: {sentiment}, Confidence: {confidence}")
        print(f"Predicted sentiment: {sentiment}, Confidence: {confidence}")
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error during prediction.'}), 500

    # Return the result
    print(f"Returning sentiment: {sentiment}")
    return jsonify({'sentiment': sentiment, 'confidence': confidence}), 200

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    print("Received /speech-to-text request.")
    
    # Check if the post request has the file part
    if 'audio_file' not in request.files:
        print("No audio file provided.")
        return jsonify({'error': 'No audio file provided.'}), 400

    audio_file = request.files['audio_file']
    filename = audio_file.filename
    print(f"Received audio file: {filename}")

    if filename == '':
        print("Empty filename received.")
        return jsonify({'error': 'No selected file.'}), 400

    # Validate MIME type
    allowed_mime_types = ['audio/wav', 'audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/mp3']
    if audio_file.mimetype not in allowed_mime_types:
        print(f"Unsupported MIME type: {audio_file.mimetype}")
        return jsonify({'error': 'Unsupported audio format.'}), 400

    # Extract file extension
    if '.' in filename:
        ext = filename.rsplit('.', 1)[-1].lower()
    else:
        # Attempt to derive extension from MIME type
        mime_to_ext = {
            'audio/wav': 'wav',
            'audio/webm': 'webm',
            'audio/ogg': 'ogg',
            'audio/mpeg': 'mp3',
            'audio/mp3': 'mp3'
        }
        ext = mime_to_ext.get(audio_file.mimetype, 'wav')  # Default to 'wav' if unknown

    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as temp_uploaded:
            temp_uploaded_path = temp_uploaded.name
            audio_file.save(temp_uploaded_path)
            print(f"Uploaded file saved as {temp_uploaded_path}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        # Convert the uploaded audio file to WAV format using pydub
        print("Converting audio file to WAV format...")
        try:
            audio = AudioSegment.from_file(temp_uploaded_path, format=ext)
            audio.export(temp_wav_path, format="wav")
            print(f"Audio file converted and saved as {temp_wav_path}")
        except Exception as conversion_error:
            print(f"Error converting audio file: {conversion_error}")
            return jsonify({'error': 'Failed to convert audio file to WAV format.'}), 400

        # Perform speech recognition
        recognizer = sr.Recognizer()
        print("Starting speech recognition...")
        try:
            with sr.AudioFile(temp_wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            print(f"Transcribed text: {text}")
            return jsonify({'text': text})
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return jsonify({'error': "Could not understand the audio."}), 400
        except sr.RequestError as e:
            print(f"Recognition service error: {e}")
            return jsonify({'error': f"Recognition service error: {e}"}), 500

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred while processing the audio file.'}), 500

    finally:
        # Clean up temporary files
        if 'temp_uploaded_path' in locals() and os.path.exists(temp_uploaded_path):
            os.remove(temp_uploaded_path)
            print(f"Temporary file {temp_uploaded_path} deleted.")
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"Temporary file {temp_wav_path} deleted.")

if __name__ == '__main__':
    # Ensure that ffmpeg is installed and accessible
    # You can install Flask and other dependencies using pip
    # Example: pip install flask pydub speechrecognition
    app.run(host='0.0.0.0', port=6000, debug=True)
