import shutil
from pathlib import Path
from typing import Any, Dict, Generator

import datasets
import pandas as pd
from constants import (
    ALL_ATTRIBUTES,
    ATTRIBUTE_DEFINITIONS,
    FEW_SHOT_PROMPT_TMPL,
    ZERO_SHOT_PROMPT_TMPL,
)
from datasets import load_dataset

RANDOM_SEED = 420


def create_zero_shot_dataset(appropriateness_dataset: datasets.DatasetDict = None):
    def generate_train_data_row(row, attribute_name: str) -> Dict[str, str]:
        return {
            "prompt": ZERO_SHOT_PROMPT_TMPL.format(
                attribute_name=attribute_name,
                attribute_definition=ATTRIBUTE_DEFINITIONS[attribute_name],
                topic=row["issue"],
                argument=row["post_text"],
                true_or_false="True" if row[attribute_name] == 1 else "False",
            ).strip()
        }

    def generate_eval_data_row(row) -> Dict[str, str]:
        result = {
            k: v for k, v in row.items() if k not in ["source_dataset", "fold0.0"]
        }
        result["correct_prediction_vector"] = [
            bool(row[attribute_name]) for attribute_name in ALL_ATTRIBUTES
        ]
        return result

    # zero-shot dataset creation
    def prompted_dataset_generator(train: bool, ds: datasets.Dataset):
        def gen():
            if train:
                for example in ds:
                    for (
                        attribute_name,
                        attribute_definition,
                    ) in ATTRIBUTE_DEFINITIONS.items():
                        yield generate_train_data_row(example, attribute_name)
            else:
                for example in ds:
                    yield generate_eval_data_row(
                        example,
                    )

        return gen

    # generate new dataset from existing and save it
    train_dataset = datasets.Dataset.from_generator(
        prompted_dataset_generator(train=True, ds=appropriateness_dataset["train"]),
        features=datasets.Features({"prompt": datasets.Value("string")}),
    )
    # generate this dataset for testing in an environment similar to the production environment
    eval_dataset = datasets.Dataset.from_generator(
        prompted_dataset_generator(
            train=False, ds=appropriateness_dataset["validation"]
        ),
    )
    # generate this dataset to evaluate while training to stop at minimal loss
    eval_dataset_prompted = datasets.Dataset.from_generator(
        prompted_dataset_generator(
            train=True, ds=appropriateness_dataset["validation"]
        ),
    )
    # create datasetdict from both datasets
    prompted_dataset = datasets.DatasetDict(
        {
            "train": train_dataset,
            "validation": eval_dataset,
            "validation_prompted": eval_dataset_prompted,
        }
    )
    # save dataset to disk
    prompted_dataset.save_to_disk("zero_shot_dataset")


"""
def create_few_shot_dataset(appropriateness_dataset: datasets.DatasetDict = None):
    df = pd.DataFrame(appropriateness_dataset["train"])

    def generate_train_data_row(row, attribute_name: str) -> Dict[str, str]:
        # sample yields a df with one row, to_dict("records")[0] transforms it to a dict
        # with a lot of bad lack, the example sample is the same as the row
        # However, the chance is so low that it is acceptable in the training set
        positive_example: dict = (
            df[df[attribute_name] == 1]
            .sample(1, random_state=RANDOM_SEED)
            .to_dict("records")[0]
        )
        negative_example: dict = (
            df[df[attribute_name] == 0]
            .sample(1, random_state=RANDOM_SEED)
            .to_dict("records")[0]
        )
        return {
            "prompt": FEW_SHOT_PROMPT_TMPL.format(
                attribute_name=attribute_name,
                attribute_definition=ATTRIBUTE_DEFINITIONS[attribute_name],
                topic=row["issue"],
                argument=row["post_text"],
                true_or_false="True" if row[attribute_name] == 1 else "False",
                first_example_topic=positive_example["issue"],
                first_example_argument=positive_example["post_text"],
                first_example_true_or_false=(
                    "True" if positive_example[attribute_name] == 1 else "False"
                ),
                second_example_topic=negative_example["issue"],
                second_example_argument=negative_example["post_text"],
                second_example_true_or_false=(
                    "True" if negative_example[attribute_name] == 1 else "False"
                ),
            ).strip()
        }

    def generate_eval_data_row(row) -> Dict[str, str]:
        result = {
            k: v for k, v in row.items() if k not in ["source_dataset", "fold0.0"]
        }
        result["correct_prediction_vector"] = [
            bool(row[attribute_name]) for attribute_name in ALL_ATTRIBUTES
        ]
        return result

    # zero-shot dataset creation
    def prompted_dataset_generator(train: bool, ds: datasets.Dataset):
        def gen():
            if train:
                for example in ds:
                    for (
                        attribute_name,
                        attribute_definition,
                    ) in ATTRIBUTE_DEFINITIONS.items():
                        yield generate_train_data_row(example, attribute_name)
            else:
                for example in ds:
                    yield generate_eval_data_row(
                        example,
                    )

        return gen

    # generate new dataset from existing and save it
    train_dataset = datasets.Dataset.from_generator(
        prompted_dataset_generator(train=True, ds=appropriateness_dataset["train"]),
        features=datasets.Features({"prompt": datasets.Value("string")}),
    )
    eval_features = {
        "issue": datasets.Value("string"),
        "post_text": datasets.Value("string"),
        "correct_prediction_vector": datasets.Sequence(datasets.Value("bool")),
    }
    for attribute in ALL_ATTRIBUTES:
        eval_features[attribute] = datasets.Value("bool")

    eval_dataset = datasets.Dataset.from_generator(
        prompted_dataset_generator(
            train=False, ds=appropriateness_dataset["validation"]
        ),
        # features=datasets.Features(eval_features),
    )
    # create datasetdict from both datasets
    prompted_dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": eval_dataset}
    )
    # save dataset to disk
    prompted_dataset.save_to_disk("few_shot_dataset")
"""

if __name__ == "__main__":
    ds = load_dataset("timonziegenbein/appropriateness-corpus")
    dirs = ["zero_shot_dataset", "few_shot_dataset"]
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

    create_zero_shot_dataset(appropriateness_dataset=ds)
    # create_few_shot_dataset(appropriateness_dataset=ds)
