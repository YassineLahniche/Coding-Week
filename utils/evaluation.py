def evaluate_model(y_true, preds, model_name):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        y_true: True labels
        preds: Predicted labels
        model_name: Name of the model being evaluated
    """
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, average="weighted")
    recall = recall_score(y_true, preds, average="weighted")
    f1 = f1_score(y_true, preds, average="weighted")

    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 30)
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
