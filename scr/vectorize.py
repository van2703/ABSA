import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# Load dataset
df = pd.read_csv("../data/absa_final.csv")

# Inputs
texts = df["text_input"]

# Labels
labels = df["sentiment"]

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(texts)

# Encode labels
le = LabelEncoder()

y = le.fit_transform(labels)

print("X shape:", X.shape)
print("Labels:", le.classes_)

# Save
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(le, "../models/label_encoder.pkl")

print("DONE")