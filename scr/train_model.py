import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs'
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# Detailed metrics
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))

# Save model
joblib.dump(model, "../models/logistic_regression.pkl")

print("MODEL SAVED")