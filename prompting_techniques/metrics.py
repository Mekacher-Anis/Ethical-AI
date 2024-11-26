import datasets
import pandas as pd
from constants import ALL_ATTRIBUTES
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(eval_dataset: datasets.Dataset, pred_dataset: pd.DataFrame):
    result = {}
    eval_dataset = pd.DataFrame(eval_dataset).set_index("post_id")
    for attribute in ALL_ATTRIBUTES:
        # create new dataframe with only the attribute columns join on post_id
        attribute_df = (
            eval_dataset[attribute]
            .to_frame()
            .join(pred_dataset[attribute].to_frame(), rsuffix="_pred")
            .dropna()
        )
        result[attribute] = metrics(
            attribute_df[attribute + "_pred"].tolist(), attribute_df[attribute].tolist()
        )
    return result


def metrics(y_pred, y_true):
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "micro_F1_score": float(f1_score(y_true, y_pred, average="micro")),
        "macro_F1_score": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": accuracy_score(y_true, y_pred),
    }
