from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

try:
    from preprocessing import clean_text
except ImportError:
    from src.preprocessing import clean_text


ASPECT_COLUMNS = [
    "chat_luong",
    "dong_goi",
    "van_chuyen",
    "gia_ca",
]


MODEL_FILENAMES = {
    "chat_luong": "lr_chat_luong.pkl",
    "dong_goi": "lr_dong_goi.pkl",
    "van_chuyen": "lr_van_chuyen.pkl",
    "gia_ca": "lr_gia_ca.pkl",
}


class ShopeeABSAPredictor:
    """
    Predictor for Shopee Aspect-Based Sentiment Analysis.

    This class loads:
    - TF-IDF vectorizer
    - 4 Logistic Regression models

    Then it predicts sentiment labels for 4 aspects:
    - chat_luong
    - dong_goi
    - van_chuyen
    - gia_ca
    """

    def __init__(self, models_dir=None):
        if models_dir is None:
            project_root = Path(__file__).resolve().parents[1]
            models_dir = project_root / "models"
        else:
            models_dir = Path(models_dir)

        self.models_dir = models_dir

        self.vectorizer_path = self.models_dir / "tfidf_vectorizer.pkl"

        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at: {self.vectorizer_path}"
            )

        self.vectorizer = joblib.load(self.vectorizer_path)

        self.models = {}

        for aspect, filename in MODEL_FILENAMES.items():
            model_path = self.models_dir / filename

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model for aspect '{aspect}' not found at: {model_path}"
                )

            self.models[aspect] = joblib.load(model_path)

    def predict_one(self, comment, return_confidence=False):
        """
        Predict sentiment labels for one comment.

        Parameters
        ----------
        comment : str
            Raw Shopee review comment.
        return_confidence : bool
            If True, return prediction confidence from Logistic Regression.

        Returns
        -------
        dict
            Prediction result for 4 aspects.
        """

        cleaned_comment = clean_text(comment)

        X_tfidf = self.vectorizer.transform([cleaned_comment])

        predictions = {}

        for aspect, model in self.models.items():
            predicted_label = model.predict(X_tfidf)[0]

            if return_confidence:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_tfidf)[0]
                    confidence = probabilities.max()

                    predictions[aspect] = {
                        "label": predicted_label,
                        "confidence": round(float(confidence), 4),
                    }
                else:
                    predictions[aspect] = {
                        "label": predicted_label,
                        "confidence": None,
                    }
            else:
                predictions[aspect] = predicted_label

        result = {
            "comment": comment,
            "clean_comment": cleaned_comment,
            "predictions": predictions,
        }

        return result

    def predict_batch(self, comments, return_confidence=False):
        """
        Predict sentiment labels for multiple comments.

        Parameters
        ----------
        comments : list[str]
            List of raw Shopee review comments.
        return_confidence : bool
            If True, return prediction confidence.

        Returns
        -------
        list[dict]
            Prediction results.
        """

        results = []

        for comment in comments:
            result = self.predict_one(
                comment=comment,
                return_confidence=return_confidence,
            )
            results.append(result)

        return results

    def predict_dataframe(
        self,
        input_csv_path,
        output_csv_path,
        text_column="comment",
    ):
        """
        Predict aspect sentiment labels for comments in a CSV file.

        Parameters
        ----------
        input_csv_path : str
            Path to input CSV file.
        output_csv_path : str
            Path to output CSV file.
        text_column : str
            Name of the column containing raw comments.
        """

        input_csv_path = Path(input_csv_path)
        output_csv_path = Path(output_csv_path)

        df = pd.read_csv(input_csv_path, encoding="utf-8-sig")

        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in input CSV file."
            )

        df["clean_comment"] = df[text_column].apply(clean_text)

        X_tfidf = self.vectorizer.transform(df["clean_comment"].fillna(""))

        for aspect, model in self.models.items():
            df[f"pred_{aspect}"] = model.predict(X_tfidf)

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            output_csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Predict aspect-based sentiment for Shopee review comments."
    )

    parser.add_argument(
        "--comment",
        type=str,
        default=None,
        help="A single Shopee review comment to predict.",
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Path to input CSV file for batch prediction.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file for batch prediction.",
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="comment",
        help="Name of the comment column in input CSV.",
    )

    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Return prediction confidence for single comment prediction.",
    )

    args = parser.parse_args()

    predictor = ShopeeABSAPredictor()

    if args.comment is not None:
        result = predictor.predict_one(
            comment=args.comment,
            return_confidence=args.confidence,
        )

        print(json.dumps(result, ensure_ascii=False, indent=4))

    elif args.input_csv is not None and args.output_csv is not None:
        output_df = predictor.predict_dataframe(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            text_column=args.text_column,
        )

        print(f"Saved predictions to: {args.output_csv}")
        print(output_df.head())

    else:
        example_comment = "Giao hàng nhanh, đóng gói kỹ nhưng sản phẩm hơi đắt"

        result = predictor.predict_one(
            comment=example_comment,
            return_confidence=True,
        )

        print("No input provided. Running example prediction:")
        print(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()