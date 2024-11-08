from typing import Any, Dict, Generator

import datasets
from datasets import load_dataset
from prompts import ATTRIBUTE_DEFINITIONS, BASE_PROMPT_TMPL


def create_zero_shot_dataset(appropriateness_dataset: datasets.Dataset = None):
    def generate_data_row(row, attribute_name: str, train: bool) -> Dict[str, str]:
        if train:
            return {
                "prompt": BASE_PROMPT_TMPL.format(
                    attribute_name="attribute_name",
                    attribute_defintion="attribute_definition",
                    topic=row["issue"],
                    argument=row["post_text"],
                    true_or_false="True" if row[attribute_name] == 1 else "False",
                ).strip()
            }
        else:
            return {
                "prompt": BASE_PROMPT_TMPL.format(
                    attribute_name="attribute_name",
                    attribute_defintion="attribute_definition",
                    topic=row["issue"],
                    argument=row["post_text"],
                    true_or_false="",
                ).strip(),
                "correct_answer": "True" if row[attribute_name] == 1 else "False",
            }

    # zero-shot dataset creation
    def prompted_dataset_generator(
        train: bool, ds: datasets.Dataset
    ) -> Generator[Dict[str, Any], None, None]:
        def gen():
            for example in ds["train"]:
                for (
                    attribute_name,
                    attribute_definition,
                ) in ATTRIBUTE_DEFINITIONS.items():
                    yield generate_data_row(example, attribute_name, train)

        return gen

    # generate new dataset from existing and save it
    train_dataset = datasets.Dataset.from_generator(
        prompted_dataset_generator(train=True, ds=appropriateness_dataset),
        features=datasets.Features({"prompt": datasets.Value("string")}),
    )
    eval_dataset = datasets.Dataset.from_generator(
        prompted_dataset_generator(train=False, ds=appropriateness_dataset),
        features=datasets.Features({
            "prompt": datasets.Value("string"),
            "correct_answer": datasets.ClassLabel(names=["True", "False"]),
        }),
    )
    # create datasetdict from both datasets
    prompted_dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": eval_dataset}
    )
    # save dataset to disk
    prompted_dataset.save_to_disk("zero_shot_dataset")


if __name__ == "__main__":
    ds = load_dataset("timonziegenbein/appropriateness-corpus")

    create_zero_shot_dataset(appropriateness_dataset=ds)
