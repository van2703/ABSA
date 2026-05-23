from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


LABEL_ORDER = ["positive", "negative", "neutral"]


def evaluate_single_aspect(
    y_true,
    y_pred,
    aspect_name,
    model_name,
    label_order=None,
):
    """
    Evaluate a single aspect classification model.
    """

    if label_order is None:
        label_order = LABEL_ORDER

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true, y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true, y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision = precision_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_order,
    )

    metrics_dict = {
        "model": model_name,
        "aspect": aspect_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    return metrics_dict, report_df, cm


def plot_confusion_matrix(cm, labels, title, save_path):
    """
    Plot and save a confusion matrix.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_all_aspects(
    y_test,
    predictions,
    aspect_columns,
    model_name,
    results_dir,
    label_order=None,
):
    """
    Evaluate all aspect models and save results.
    """

    if label_order is None:
        label_order = LABEL_ORDER

    results_dir = Path(results_dir)
    reports_dir = results_dir / "classification_reports"
    confusion_dir = results_dir / "confusion_matrices"

    reports_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for aspect in aspect_columns:
        y_true = y_test[aspect]
        y_pred = predictions[aspect]

        metrics_dict, report_df, cm = evaluate_single_aspect(
            y_true=y_true,
            y_pred=y_pred,
            aspect_name=aspect,
            model_name=model_name,
            label_order=label_order,
        )

        all_results.append(metrics_dict)

        report_path = reports_dir / f"{model_name.lower().replace(' ', '_')}_{aspect}_classification_report.csv"
        report_df.to_csv(report_path, encoding="utf-8-sig")

        cm_path = confusion_dir / f"{model_name.lower().replace(' ', '_')}_{aspect}_confusion_matrix.png"
        plot_confusion_matrix(
            cm=cm,
            labels=label_order,
            title=f"{model_name} Confusion Matrix - {aspect}",
            save_path=cm_path,
        )

    results_df = pd.DataFrame(all_results)

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    results_df[metric_columns] = results_df[metric_columns].round(4)

    return results_df