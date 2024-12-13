from pathlib import Path
import pandas as pd

pd.options.mode.chained_assignment = None

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

    num_labels_per_annotator = 60
    annotators = ["Anis", "Linus", "Julian", "Erik", "Peer"]
    annotators_per_sample = 3

    num_labels = num_labels_per_annotator * len(annotators) // annotators_per_sample
    print(f"Number of annotated samples: {num_labels}")

    for set in ["test"]: #, "train", "valid"]:
        full_df = pd.read_csv(root_dir / f"{set}.csv")
        df = full_df[full_df["Inappropriateness"] == 1]
        print(f"{set} set: num total samples: {len(full_df)}, num inappropriate samples: {len(df)}")
        df = df.iloc[:num_labels]

        final_df = df[["post_text", "issue"]]
        final_df["document_labels"] = df.aggregate(aggregate_labels, axis=1)

        indices = pd.Series([], dtype=int)

        for i, annotator in enumerate(annotators):
            lower_bound = i * (num_labels_per_annotator // annotators_per_sample)
            if (lower_bound + num_labels_per_annotator) > num_labels:
                last_items = final_df.iloc[lower_bound:]
                first_items = final_df.iloc[:(lower_bound + num_labels_per_annotator) % num_labels]
                annotator_df = pd.concat([last_items, first_items])
            else:
                annotator_df = final_df.iloc[lower_bound : (lower_bound + num_labels_per_annotator)]
            print(f"{annotator}: {lower_bound=}, upper_bound={(lower_bound + num_labels_per_annotator) % num_labels}, {len(annotator_df)}")
            indices = pd.concat([indices, annotator_df.index.to_series()])
            annotator_df.to_json(result_dir / f"{annotator}_{set}_{num_labels_per_annotator}_{annotators_per_sample}.jsonl", orient='records', lines=True, index=True)

        index_counts = indices.value_counts()
        print(f"Indices with wrong number of annotations: {index_counts[index_counts != annotators_per_sample].to_dict()}")

        #final_df["labeled_post_text"] = final_df.aggregate(combine_text_and_labels, axis=1)
        #final_df.to_json(result_dir / f"{set}_{num_labels_per_annotator}_{annotators_per_sample}.jsonl", orient='records', lines=True)
