from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_auroc(X, y, classifier, is_binary):
    validation_split = len(X) // 2

    classifier.fit(X[:validation_split].cpu(), y[:validation_split].cpu())
    if is_binary:
        probs = classifier.predict_proba(X[validation_split:].cpu())[:, 1]
    else:
        probs = classifier.predict_proba(X[validation_split:].cpu())
    auprc = average_precision_score(y[validation_split:].cpu().numpy(), probs)
    auroc = roc_auc_score(y[validation_split:].cpu().numpy(), probs)

    return auprc, auroc
