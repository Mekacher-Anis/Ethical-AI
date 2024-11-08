from typing import Any, Dict, Generator

import datasets

from prompts import *
from datasets import load_dataset


if __name__ == "__main__":
    ds = load_dataset("timonziegenbein/appropriateness-corpus")

    # zero-shot dataset creation
    def prompted_dataset_generator() -> Generator[Dict[str, Any], None, None]:
        for example in ds["train"]:
            for attribute_name, attribute_definition in ATTRIBUTE_DEFINITIONS.items():
                yield {
                    "prompt": BASE_PROMPT_TMPL.format(
                        attribute_name=attribute_name,
                        attribute_defintion=attribute_definition,
                        topic=example["issue"],
                        argument=example["post_text"],
                        true_or_false=(
                            "True" if example[attribute_name] == 1 else "False"
                        ),
                    ).strip()
                }

    # generate new dataset from existing and save it
    new_dataset = datasets.Dataset.from_generator(prompted_dataset_generator)
    new_dataset.save_to_disk("prompted_dataset")
    print(new_dataset)
