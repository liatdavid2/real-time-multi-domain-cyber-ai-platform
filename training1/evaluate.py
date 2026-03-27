from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


def evaluate_model(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True)

    print("Classification report:")
    print(classification_report(y_true, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        # positive class (attack = 1)
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],

        # additional useful metrics
        "f1_weighted": report["weighted avg"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "support_attack": report["1"]["support"]
    }

    return metrics