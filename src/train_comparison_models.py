from pathlib import Path

import joblib
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


ASPECT_COLUMNS = [
    "chat_luong",
    "dong_goi",
    "van_chuyen",
    "gia_ca",
]


def evaluate_single_aspect(y_true, y_pred, aspect_name, model_name):
    """
    Evaluate one model on one aspect.
    """

    return {
        "model": model_name,
        "aspect": aspect_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def train_and_evaluate_model(
    model,
    model_name,
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test,
    models_dir,
):
    """
    Train one type of model for all four aspects.
    """

    results = []

    safe_model_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    for aspect in ASPECT_COLUMNS:
        print(f"Training {model_name} for aspect: {aspect}")

        current_model = model.__class__(**model.get_params())

        current_model.fit(X_train_tfidf, y_train[aspect])

        y_pred = current_model.predict(X_test_tfidf)

        result = evaluate_single_aspect(
            y_true=y_test[aspect],
            y_pred=y_pred,
            aspect_name=aspect,
            model_name=model_name,
        )

        results.append(result)

        model_path = models_dir / f"{safe_model_name}_{aspect}.pkl"
        joblib.dump(current_model, model_path)

        print(f"Saved model to: {model_path}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print("-" * 80)

    return results


def create_model_comparison_table(comparison_all_metrics):
    """
    Create a wide comparison table using Macro F1-score.
    """

    macro_f1_pivot = comparison_all_metrics.pivot_table(
        index="model",
        columns="aspect",
        values="macro_f1",
    ).reset_index()

    model_comparison_results = macro_f1_pivot.rename(columns={
        "chat_luong": "chat_luong_macro_f1",
        "dong_goi": "dong_goi_macro_f1",
        "van_chuyen": "van_chuyen_macro_f1",
        "gia_ca": "gia_ca_macro_f1",
    })

    macro_f1_columns = [
        "chat_luong_macro_f1",
        "dong_goi_macro_f1",
        "van_chuyen_macro_f1",
        "gia_ca_macro_f1",
    ]

    model_comparison_results["average_macro_f1"] = (
        model_comparison_results[macro_f1_columns].mean(axis=1)
    )

    model_comparison_results = model_comparison_results.sort_values(
        by="average_macro_f1",
        ascending=False,
    )

    return model_comparison_results


def main(run_random_forest=False):
    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    results_dir = project_root / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    lr_results_path = results_dir / "logistic_regression_results.csv"

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")

    X_train = train_df["clean_comment"].fillna("")
    X_test = test_df["clean_comment"].fillna("")

    y_train = train_df[ASPECT_COLUMNS]
    y_test = test_df[ASPECT_COLUMNS]

    vectorizer = joblib.load(vectorizer_path)

    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    all_results = []

    # Load Logistic Regression results from Phase 7
    if lr_results_path.exists():
        lr_results_df = pd.read_csv(lr_results_path, encoding="utf-8-sig")
        all_results.extend(lr_results_df.to_dict(orient="records"))
    else:
        print("Warning: Logistic Regression results file not found.")

    # Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_results = train_and_evaluate_model(
        model=nb_model,
        model_name="Multinomial Naive Bayes",
        X_train_tfidf=X_train_tfidf,
        X_test_tfidf=X_test_tfidf,
        y_train=y_train,
        y_test=y_test,
        models_dir=models_dir,
    )
    all_results.extend(nb_results)

    # Linear SVM
    svm_model = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000,
    )
    svm_results = train_and_evaluate_model(
        model=svm_model,
        model_name="Linear SVM",
        X_train_tfidf=X_train_tfidf,
        X_test_tfidf=X_test_tfidf,
        y_train=y_train,
        y_test=y_test,
        models_dir=models_dir,
    )
    all_results.extend(svm_results)

    # Random Forest optional
    if run_random_forest:
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_results = train_and_evaluate_model(
            model=rf_model,
            model_name="Random Forest",
            X_train_tfidf=X_train_tfidf,
            X_test_tfidf=X_test_tfidf,
            y_train=y_train,
            y_test=y_test,
            models_dir=models_dir,
        )
        all_results.extend(rf_results)

    comparison_all_metrics = pd.DataFrame(all_results)

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    comparison_all_metrics[metric_columns] = (
        comparison_all_metrics[metric_columns].round(4)
    )

    comparison_all_metrics_path = results_dir / "comparison_all_metrics.csv"
    comparison_all_metrics.to_csv(
        comparison_all_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved all comparison metrics to: {comparison_all_metrics_path}")

    model_comparison_results = create_model_comparison_table(
        comparison_all_metrics
    )

    model_comparison_results = model_comparison_results.round(4)

    model_comparison_path = results_dir / "model_comparison_results.csv"
    model_comparison_results.to_csv(
        model_comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved model comparison results to: {model_comparison_path}")


if __name__ == "__main__":
    main(run_random_forest=True)