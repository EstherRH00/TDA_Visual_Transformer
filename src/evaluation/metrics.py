from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "f1": f1_score(y_true, y_pred_bin),
        "auc": roc_auc_score(y_true, y_pred)
    }