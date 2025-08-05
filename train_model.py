# train_model.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Step 1: Load preprocessed data and vectorizer from the 'data' and 'models' folders
try:
    X_train_vectorized = joblib.load("data/X_train_vectorized.pkl")
    X_test_vectorized = joblib.load("data/X_test_vectorized.pkl")
    y_train = joblib.load("data/y_train.pkl")
    y_test = joblib.load("data/y_test.pkl")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run 'python preprocess.py' first to create the necessary files.")
    exit()

# Step 2: Encode string labels to numbers for the model to use
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Save the encoder for decoding predictions later
joblib.dump(le, "models/label_encoder.pkl")

# Step 3: Initialize and train the classifier
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train_vectorized, y_train_encoded)

# Step 4: Evaluate the model
y_pred_encoded = model.predict(X_test_vectorized)
print("📊 Classification Report:\n")
print(classification_report(y_test_encoded, y_pred_encoded, target_names=le.classes_))

# Step 5: Save the trained model
joblib.dump(model, "models/sentiment_model.pkl")

print("✅ Model training complete. Saved to models/sentiment_model.pkl")