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
    # quantization_config = custom_quantization_config or BitsAndBytesConfig(
    #    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    # )
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
            # config=quantization_config,
            torch_dtype="float16",
        )
    if load_only_tokenizer:
        return tokenizer
    elif load_only_model:
        return model
    else:
        return tokenizer, model
