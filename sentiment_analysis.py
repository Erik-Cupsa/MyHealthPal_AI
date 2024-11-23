# sentiment_analysis_balanced.py

import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
DATA_PATH = 'sentiment140.csv'
SAMPLE_SIZE = 100000  # Adjust based on your system's capacity
MAX_NUM_WORDS = 20000  # Vocabulary size
MAX_SEQUENCE_LENGTH = 100  # Maximum length of each text sequence
EMBEDDING_DIM = 100  # Embedding dimensions
BATCH_SIZE = 512
EPOCHS = 20

# Debug: Print each line of the CSV as it is read
print("Reading and processing the CSV file line by line...")
with open(DATA_PATH, 'r', encoding='latin-1') as file:
    for i, line in enumerate(file):
        print(f"Line {i + 1}: {line.strip()}")
        # Limit the number of lines printed for large files
        if i >= 9:  # Stop after 10 lines to avoid overwhelming output
            print("...skipping the rest for brevity.")
            break

# Load the dataset with correct parameters
print("\nLoading the dataset into a pandas DataFrame...")
data = pd.read_csv(
    DATA_PATH,
    encoding='latin-1',  # Matches the file encoding
    header=None,         # No headers in the file
    names=['target', 'ids', 'date', 'flag', 'user', 'text'],  # Column names
    quoting=0,           # Ensure quotes are handled correctly
    sep=',',             # Fields are comma-separated
    engine='python',     # Use the Python engine for complex cases
)

# Optionally, sample a subset to reduce memory usage
print(f"\nSampling {SAMPLE_SIZE} samples from the dataset...")
data = data.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# Map target values to sentiments
def map_target(value):
    if value == 0:
        return 'sad'
    elif value == 2:
        return 'neutral'
    elif value == 4:
        return 'happy'

print("Mapping target values to sentiment labels...")
data['sentiment'] = data['target'].apply(map_target)

# Check class distribution before balancing
print("\nClass distribution before balancing:")
print(data['sentiment'].value_counts())

# Find the minimum class count
min_count = data['sentiment'].value_counts().min()
print(f"\nMinimum class count: {min_count}")

# Perform undersampling to balance the classes
balanced_data = data.groupby('sentiment').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)

# Verify the new class distribution
print("\nClass distribution after balancing:")
print(balanced_data['sentiment'].value_counts())

# Preprocessing function
def preprocess_text(text):
    # Remove URLs, mentions, hashtags, punctuation, and digits
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    # Lowercase the text
    text = text.lower()
    return text.strip()

print("\nApplying preprocessing to the 'text' column...")
balanced_data['clean_text'] = balanced_data['text'].apply(preprocess_text)

# Debug: Print a few rows to verify the DataFrame content
print("\nSample rows from the preprocessed DataFrame:")
print(balanced_data.head())

# Split the data
print("\nSplitting the data into training and testing sets...")
X = balanced_data['clean_text']
y = balanced_data['sentiment']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Ensure that label encoding is correct
print("\nLabel encoding mapping:")
for i, class_label in enumerate(label_encoder.classes_):
    print(f"{i}: {class_label}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Tokenize and pad sequences
print("\nTokenizing and padding text data...")
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(
    X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
)
X_test_padded = pad_sequences(
    X_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
)

# Save the tokenizer for future use
joblib.dump(tokenizer, 'tokenizer.joblib')
print("\nTokenizer saved as 'tokenizer.joblib'.")

# Build the neural network model
print("\nBuilding the neural network model...")
model = Sequential([
    Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: sad, neutral, happy
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Define early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with multiple epochs
print("\nTraining the model...")
history = model.fit(
    X_train_padded,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the model
print("\nEvaluating the model on the test set...")
y_pred = model.predict(X_test_padded)
y_pred_classes = y_pred.argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Save the trained model
print("\nSaving the trained model to 'sentiment_model.h5'...")
model.save('sentiment_model.h5')
print("Model saved successfully.")

# Save the label encoder for future use
joblib.dump(label_encoder, 'label_encoder.joblib')
print("Label encoder saved as 'label_encoder.joblib'.")

print("\nScript completed successfully.")
