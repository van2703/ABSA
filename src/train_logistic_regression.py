from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ASPECT_COLUMNS = [
    "chat_luong",
    "dong_goi",
    "van_chuyen",
    "gia_ca",
]


def train_logistic_regression_models(
    train_path: str,
    test_path: str,
    models_dir: str,
):
    """
    Train four Logistic Regression models for four ABSA aspects.

    Parameters
    ----------
    train_path : str
        Path to train.csv.
    test_path : str
        Path to test.csv.
    models_dir : str
        Directory to save TF-IDF vectorizer and trained models.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    models : dict
        Dictionary of trained Logistic Regression models.
    predictions : dict
        Dictionary of predictions on the test set.
    """

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")

    X_train = train_df["clean_comment"].fillna("")
    X_test = test_df["clean_comment"].fillna("")

    y_train = train_df[ASPECT_COLUMNS]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved TF-IDF vectorizer to: {vectorizer_path}")
    print("X_train_tfidf shape:", X_train_tfidf.shape)
    print("X_test_tfidf shape:", X_test_tfidf.shape)

    models = {}
    predictions = {}

    for aspect in ASPECT_COLUMNS:
        print(f"\nTraining Logistic Regression model for aspect: {aspect}")

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )

        model.fit(X_train_tfidf, y_train[aspect])

        y_pred = model.predict(X_test_tfidf)

        models[aspect] = model
        predictions[aspect] = y_pred

        model_path = models_dir / f"lr_{aspect}.pkl"
        joblib.dump(model, model_path)

        print(f"Saved model to: {model_path}")
        print("Classes:", model.classes_)

    return vectorizer, models, predictions


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    train_path = project_root / "data" / "processed" / "train.csv"
    test_path = project_root / "data" / "processed" / "test.csv"
    models_dir = project_root / "models"

    train_logistic_regression_models(
        train_path=str(train_path),
        test_path=str(test_path),
        models_dir=str(models_dir),
    )