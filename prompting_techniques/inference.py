import os

import datasets
from constants import (ATTRIBUTE_DEFINITIONS, ATTRIBUTE_RELATIONS,
                       BASE_PROMPT_TMPL)
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from sklearn.metrics import accuracy_score 


def test_model(model, tokenizer, eval_dataset):

    # get token id for "true" and "false" tokens
    true_token_id = tokenizer.encode("true", return_tensors="pt").to("cuda")[0][1]
    false_token_id = tokenizer.encode("false", return_tensors="pt").to("cuda")[0][1]

    results = []
    for i, sample in tqdm(enumerate(eval_dataset)):
        cur_row = {"post_id": sample["post_id"]}

        for attribute_name, attribute_definition in ATTRIBUTE_DEFINITIONS.items():
            prompt = BASE_PROMPT_TMPL.format(
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
            true_logit = model_output["logits"][0][0][true_token_id]
            false_logit = model_output["logits"][0][0][false_token_id]
            cur_row[attribute_name] = true_logit > false_logit
        for attribute_name, attribute_relation in ATTRIBUTE_RELATIONS.items():
            cur_row[attribute_name] = any(
                [cur_row[attribute] for attribute in attribute_relation]
            )
        results.append(cur_row)
    return results


def load_untrained_llama2_model():
    load_dotenv()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    )
    # pull llama2 model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        token=os.getenv("HF_TOKEN"),
        device_map="auto",
        config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        token=os.getenv("HF_TOKEN"),
        device_map="auto",
        config=quantization_config,
        torch_dtype="float16",
    )
    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_untrained_llama2_model()
    print("Model loaded")
    # load dataset from file
    zero_shot_dataset = datasets.load_from_disk("zero_shot_dataset")
    eval_dataset = zero_shot_dataset["validation"]
    results = test_model(model, tokenizer, eval_dataset)
    print(results)
    print("Accuracy:", accuracy_score(eval_dataset["Inappropriateness"], results["Inappropriateness"]))
