
from sklearn.ensemble import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, f1_score
import inference
import datasets
#from skllm.preprocessing import GPTVectorizer
#from skllm import ZeroShotGPTClassifier
import pandas as pd

def min_recall_precision(y_true, y_pred, sample_weight=None):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)

def metrics(y_pred, y_true):
    data={
        "min_racall_precision": min_recall_precision(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "micro_F1_score": f1_score(y_true, y_pred, average='micro'),
        "Accuracy": accuracy_score(y_true, y_pred)
    }
    
    
    return data

if __name__ == "__main__":
    tokenizer, model =inference.load_untrained_llama2_model()
    print("Model loaded")
    zero_shot_dataset = datasets.load_from_disk("zero_shot_dataset")
    eval_dataset = zero_shot_dataset["validation"]
    print("Dataset loaded")
    y_pred=inference.test_model(model, tokenizer, eval_dataset)
    print(pd.DataFrame(metrics(y_pred, eval_dataset)))
    