import os
from typing import Tuple, Union

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_untrained_llama2_model(
    custom_quantization_config: BitsAndBytesConfig = None,
    load_only_tokenizer: bool = False,
    load_only_model: bool = False,
) -> Union[
    AutoTokenizer, AutoModelForCausalLM, Tuple[AutoTokenizer, AutoModelForCausalLM]
]:
    load_dotenv()
    assert not (
        load_only_tokenizer and load_only_model
    ), "Cannot load only tokenizer and only model at the same time"
    load_both = not load_only_tokenizer and not load_only_model
    if load_only_tokenizer or load_both:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            token=os.getenv("HF_TOKEN"),
            device_map="cuda",
        )
        tokenizer.pad_token = tokenizer.eos_token
    if load_only_model or load_both:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            token=os.getenv("HF_TOKEN"),
            device_map="cuda",
            config=custom_quantization_config,
            torch_dtype="float16",
        )
    if load_only_tokenizer:
        assert tokenizer is not None
        return tokenizer
    elif load_only_model:
        assert model is not None
        return model
    else:
        assert tokenizer is not None and model is not None
        return tokenizer, model
