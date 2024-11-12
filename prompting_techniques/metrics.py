import sklearn
from sklearn.ensemble import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
import inference

def min_recall_precision(est, X, y_true, sample_weight=None):
    y_pred = est.predict(X)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)

grid = GridSearchCV(
    estimator=inference.load_untrained_llama2_model(), param_grid={'class_weight'}
    #WIP
)
    