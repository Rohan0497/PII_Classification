import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools
from evaluate import load

def load_labels():
    with open('./data/label2id.json', 'r') as f:
        label2id = json.load(f)
    return label2id

def compute_metrics(p):
    label2id = load_labels()
    id2label = {idx: label for label, idx in label2id.items()}
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [id2label[pred] for pred_list, label_list in zip(predictions, labels) for pred, label in zip(pred_list, label_list) if label != -100]
    true_labels = [id2label[label] for label_list, pred_list in zip(labels, predictions) for label, pred in zip(label_list, pred_list) if label != -100]
    
    metric = load("seqeval") 
    results = metric.compute(predictions=[true_predictions], references=[true_labels])
    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)  # Handling zero division
       
    # Save the classification report
    with open('./output_dir/classification_report.json', 'w') as f:
        json.dump(report, f)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results.get("overall_accuracy", 0),
    }
