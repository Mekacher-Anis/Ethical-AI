import shutil
from pathlib import Path
from typing import Any, Dict, Generator

import datasets
from constants import ALL_ATTRIBUTES, ATTRIBUTE_DEFINITIONS, BASE_PROMPT_TMPL
from datasets import load_dataset


def create_zero_shot_dataset(appropriateness_dataset: datasets.Dataset = None):
    def generate_train_data_row(row, attribute_name: str) -> Dict[str, str]:
        return {
            "prompt": BASE_PROMPT_TMPL.format(
                attribute_name="attribute_name",
                attribute_definition="attribute_definition",
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
    def prompted_dataset_generator(
        train: bool, ds: datasets.Dataset
    ) -> Generator[Dict[str, Any], None, None]:
        def gen():
            if train:
                for example in ds["train"]:
                    for (
                        attribute_name,
                        attribute_definition,
                    ) in ATTRIBUTE_DEFINITIONS.items():
                        yield generate_train_data_row(example, attribute_name)
            else:
                for example in ds["validation"]:
                    yield generate_eval_data_row(
                        example,
                    )

        return gen

    # generate new dataset from existing and save it
    train_dataset = datasets.Dataset.from_generator(
        prompted_dataset_generator(train=True, ds=appropriateness_dataset),
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
        prompted_dataset_generator(train=False, ds=appropriateness_dataset),
        # features=datasets.Features(eval_features),
    )
    # create datasetdict from both datasets
    prompted_dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": eval_dataset}
    )
    # save dataset to disk
    prompted_dataset.save_to_disk("zero_shot_dataset")


if __name__ == "__main__":
    ds = load_dataset("timonziegenbein/appropriateness-corpus")
    if Path("zero_shot_dataset").exists():
        shutil.rmtree("zero_shot_dataset")

    create_zero_shot_dataset(appropriateness_dataset=ds)
