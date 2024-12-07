from pathlib import Path
import pandas as pd


classes = [ # class layer
    #'Inappropriateness', # 0
    
    #'Toxic Emotions', # 1
    'Excessive Intensity', # 2
    'Emotional Deception', # 2
    
    #'Missing Commitment', # 1
    'Missing Seriousness', # 2
    'Missing Openness', # 2
    
    #'Missing Intelligibility', # 1
    'Unclear Meaning', # 2
    'Missing Relevance', # 2
    'Confusing Reasoning', # 2
    
    #'Other Reasons', # 1
    'Detrimental Orthography', # 2
    'Reason Unclassified', # 2
]

def aggregate_labels(x):
    labels = []
    for c in classes:
        if x[c]:
            labels.append(c)
    return labels

def combine_text_and_labels(x):
    text = f"{x["label"]}:\n{x["post_text"]}"
    return text

if __name__ == "__main__":
    root_dir = Path("../datasets/appropriateness-corpus")
    result_dir = Path("./doccano")
    for set in ["test", "train", "valid"]:
        df = pd.read_csv(root_dir / f"{set}.csv")
        final_df = df[["post_text"]]
        final_df["document_labels"] = df.aggregate(aggregate_labels, axis=1)
        #final_df["labeled_post_text"] = final_df.aggregate(combine_text_and_labels, axis=1)
        final_df.to_json(result_dir / f"{set}.jsonl", orient='records', lines=True)