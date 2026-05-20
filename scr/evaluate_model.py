import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv("../data/absa_final.csv")

# Split dataframe first
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=2
)

X_train = vectorizer.fit_transform(train_df["text_input"])
X_test = vectorizer.transform(test_df["text_input"])

# Encode labels
le = LabelEncoder()

y_train = le.fit_transform(train_df["sentiment"])
y_test = le.transform(test_df["sentiment"])

# Train
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs'
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Save predictions
test_df["predicted"] = le.inverse_transform(y_pred)

# Wrong predictions
wrong_df = test_df[
    test_df["sentiment"] != test_df["predicted"]
]
feature_names = vectorizer.get_feature_names_out()

for i, class_name in enumerate(le.classes_):

    top10 = model.coef_[i].argsort()[-10:]

    print(f"\nCLASS: {class_name}")

    for idx in top10:
        print(feature_names[idx])

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_)
os.makedirs("../report", exist_ok=True)
plt.savefig("../report/confusion_matrix.png")
print(wrong_df.head(20))