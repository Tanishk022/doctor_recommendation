import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("firebaseKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Fetch data from Firestore
docs = db.collection("doctors").stream()
data = [doc.to_dict() for doc in docs]
df = pd.DataFrame(data)

# Preprocess function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return " ".join(text.lower().split())

# Create combined field
df['combined_text'] = (df['specialization'].fillna('') + " " + df['Description'].fillna('')).apply(preprocess_text)

# Train TF-IDF model
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df['combined_text'])

# Save processed data and model
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(vectors, "vectors.pkl")
df.to_csv("processed_doctor_data.csv", index=False)

print("✅ Training complete and files saved.")
