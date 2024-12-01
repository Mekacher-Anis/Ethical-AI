import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support


# define which metrics to compute for evaluation
def compute_metrics(p, id2class: dict[int, str], id_mappings: dict[int, list[int]] = None):
    predictions, labels = p
    predictions_binary = predictions > 0

    if id_mappings:
        inferred_preds = np.zeros((predictions_binary.shape[0], len(id_mappings)))
        inferred_labels = np.zeros((labels.shape[0], len(id_mappings)))
        inferred_preds[:, :predictions_binary.shape[1]] = predictions_binary
        for id, mappings in id_mappings.items():
            inferred_preds[:, id] = np.max(inferred_preds[:, mappings], axis=1)
            inferred_labels[:, id] = np.max(inferred_labels[:, mappings], axis=1)
    else:
        inferred_preds = predictions_binary
        inferred_labels = labels
    
    # outdated, does not fit metric calculations from paper baseline
    """
    # Average metrics
    f1_micro = f1_score(inferred_labels, inferred_preds, average='micro', zero_division=0.0)
    f1_macro = f1_score(inferred_labels, inferred_preds, average='macro', zero_division=0.0)
    f1_weighted = f1_score(inferred_labels, inferred_preds, average='weighted', zero_division=0.0)
    
    precision_micro = precision_score(inferred_labels, inferred_preds, average='micro', zero_division=0.0)
    precision_macro = precision_score(inferred_labels, inferred_preds, average='macro', zero_division=0.0)
    precision_weighted = precision_score(inferred_labels, inferred_preds, average='weighted', zero_division=0.0)
    
    recall_micro = recall_score(inferred_labels, inferred_preds, average='micro', zero_division=0.0)
    recall_macro = recall_score(inferred_labels, inferred_preds, average='macro', zero_division=0.0)
    recall_weighted = recall_score(inferred_labels, inferred_preds, average='weighted', zero_division=0.0)
    
    # Per-class metrics
    precision_per_id = precision_score(inferred_labels, inferred_preds, average=None, zero_division=0.0)
    recall_per_id = recall_score(inferred_labels, inferred_preds, average=None, zero_division=0.0)
    f1_per_id = f1_score(inferred_labels, inferred_preds, average=None, zero_division=0.0)

    precision_per_class = {f"precision_{id2class[id]}": value for id, value in enumerate(precision_per_id)}
    recall_per_class = {f"recall_{id2class[id]}": value for id, value in enumerate(recall_per_id)}
    f1_per_class = {f"f1_{id2class[id]}": value for id, value in enumerate(f1_per_id)}
    """
    
    classwise_metrics = {}
    average = "macro"

    summed_precision = 0
    summed_recall = 0
    summed_f1 = 0
    for id, class_label in id2class.items():
        scores = precision_recall_fscore_support(
            [x[id] for x in inferred_labels], [x[id] for x in inferred_preds], average=average, zero_division=0.0
        )

        classwise_metrics[f"precision_{class_label}"] = scores[0]
        summed_precision += scores[0]
        classwise_metrics[f"recall_{class_label}"] = scores[1]
        summed_recall += scores[1]
        classwise_metrics[f"f1_{class_label}"] = scores[2]
        summed_f1 += scores[2]
    
    return {
        'f1_macro': summed_f1 / len(id2class),
        'precision_macro': summed_precision / len(id2class),
        'recall_macro': summed_recall / len(id2class),
        
        **classwise_metrics,
    }