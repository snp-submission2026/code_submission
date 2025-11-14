from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_predictions(y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm
