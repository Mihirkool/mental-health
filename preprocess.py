# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define the path to your raw dataset
file_path = "data/MH_Campaigns1723.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the file is in the 'data' directory.")
    exit()

# Use a random sample of the data for faster processing
df_sample = df.sample(n=10000, random_state=42).reset_index(drop=True)
print(f"Using a sample of {len(df_sample)} rows for faster processing.")

# Extract features (tweet text) and targets (campaign labels)
X = df_sample['tweet'].astype(str)
y = df_sample['campaign']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Save the processed data and the vectorizer to disk
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(X_train_vectorized, "data/X_train_vectorized.pkl")
joblib.dump(y_train, "data/y_train.pkl")
joblib.dump(X_test_vectorized, "data/X_test_vectorized.pkl")
joblib.dump(y_test, "data/y_test.pkl")

print("✅ Preprocessing complete. Data and vectorizer saved successfully.")