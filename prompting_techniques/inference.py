import json

import datasets
import pandas as pd
import torch
from constants import ATTRIBUTE_DEFINITIONS, ATTRIBUTE_RELATIONS, ZERO_SHOT_PROMPT_TMPL
from tqdm import tqdm
from utils import load_untrained_llama2_model

from prompting_techniques.metrics import calculate_metrics


def test_model_zero_shot(model, tokenizer, eval_dataset):

    # get token id for "true" and "false" tokens
    true_token_id = tokenizer.encode("true", return_tensors="pt").to("cuda")[0][1]
    false_token_id = tokenizer.encode("false", return_tensors="pt").to("cuda")[0][1]

    results = []
    for i, sample in tqdm(enumerate(eval_dataset)):
        cur_row = {"post_id": sample["post_id"]}

        for attribute_name, attribute_definition in ATTRIBUTE_DEFINITIONS.items():
            prompt = ZERO_SHOT_PROMPT_TMPL.format(
                attribute_name=attribute_name,
                attribute_definition=attribute_definition,
                topic=sample["issue"],
                argument=sample["post_text"],
                true_or_false="",
            )
            tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            # get probability distribution vector of next token from LLM
            model_output = model.generate(
                tokens,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_logits=True,
            )
            # first batch, first token of sequence
            true_logit = model_output["logits"][0][0][true_token_id]
            false_logit = model_output["logits"][0][0][false_token_id]
            is_attribute_true: torch.Tensor = true_logit > false_logit
            cur_row[attribute_name] = 1 if is_attribute_true.item() else 0
        for attribute_name, attribute_relation in ATTRIBUTE_RELATIONS.items():
            cur_row[attribute_name] = (
                1
                if any([cur_row[attribute] for attribute in attribute_relation])
                else 0
            )
        results.append(cur_row)
    results = pd.DataFrame(results).set_index("post_id")
    return results


if __name__ == "__main__":
    # load llama2 tokenizer
    tokenizer, model = load_untrained_llama2_model()
    model.load_adapter("output/checkpoint-300")
    print("Model loaded")
    # load dataset from file
    zero_shot_dataset = datasets.load_from_disk("zero_shot_dataset")
    eval_dataset = zero_shot_dataset["validation"]
    results = test_model_zero_shot(model, tokenizer, eval_dataset)
    metrics = calculate_metrics(eval_dataset, results)
    with open("zero_shot_with_tuning.json", "w") as f:
        json.dump(metrics, f, indent=4)
