import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def compute_accuracy(preds, labels):
    return accuracy_score(labels, preds)


def compute_f1(preds, labels, average="macro"):
    return f1_score(labels, preds, average=average)


def compute_confusion_matrix(preds, labels):
    return confusion_matrix(labels, preds)


def classification_report_to_csv(preds, labels, class_names, filepath):
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(filepath, index=True)
