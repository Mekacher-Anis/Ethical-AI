from typing import List, Optional, Tuple, Dict
import torch
import outlines
from enum import Enum
from pydantic import BaseModel, constr, conint
from dotenv import load_dotenv
import os

load_dotenv()
model = outlines.models.transformers(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16, "token": os.getenv("HF_TOKEN")},
    device="cuda",
)

class Severity(str, Enum):
    minimal = "minimal"
    significant = "significant"


class ProblematicSnippet(BaseModel):
    original_text: str
    severity: Severity
    start: Optional[conint(ge=0)] = None
    end: Optional[conint(ge=0)] = None


class TextAnalysis(BaseModel):
    problematic_snippets: List[ProblematicSnippet]


class Token(BaseModel):
    text: str
    start: conint(ge=0)
    end: conint(ge=0)
    severity: Optional[Severity] = None


generator = outlines.generate.json(
    model, TextAnalysis, sampler=outlines.samplers.multinomial(temperature=0.1)
)

prompt_template = """
{attribute_definition}
You are provided with a comment which contains {attribute}. Output the snippets which lead to this classification.
----------
Comment in question:
{comment}
"""


def generate_text_analysis(attribute_definition: str, attribute: str, comment: str):
    return generator(
        prompt_template.format(attribute_definition=attribute_definition, attribute=attribute, comment=comment))
from datasets import load_dataset

ds = load_dataset("timonziegenbein/appropriateness-corpus")["test"]
import re
from constants import ATTRIBUTE_DEFINITIONS


def analyse_post_for_attribute(post_text: str, attribute_name: str, attribute_definition: str) -> List[
    Tuple[str, float]]:
    analysis_result = generate_text_analysis(attribute_name, attribute_definition, post_text)
    update_snippet_positions(analysis_result.problematic_snippets, post_text)
    token_list = tokenize_post_text(post_text)
    assign_severity_to_tokens(token_list, analysis_result.problematic_snippets)
    return convert_tokens_to_output(token_list)


def update_snippet_positions(snippets: List[ProblematicSnippet], post_text: str) -> None:
    for snippet in snippets:
        re_result = re.search(snippet.original_text, post_text)
        if re_result is None:
            continue
        snippet.start = re_result.start()
        snippet.end = re_result.end()
    snippets[:] = [snippet for snippet in snippets if snippet.start is not None]


def tokenize_post_text(post_text: str) -> List[Token]:
    tokenized_text = model.tokenizer.encode(post_text)[0][0]
    seek_start = 0
    token_list = []
    for token_id in tokenized_text:
        text = model.tokenizer.decode([token_id])[0]
        if text:
            token = Token(text=text, start=seek_start, end=seek_start + len(text))
            seek_start += len(text)
            token_list.append(token)
    return token_list


def assign_severity_to_tokens(token_list: List[Token], snippets: List[ProblematicSnippet]) -> None:
    for token in token_list:
        for snippet in snippets:
            if snippet.start <= token.start and token.end <= snippet.end:
                token.severity = snippet.severity


def convert_tokens_to_output(token_list: List[Token]) -> List[Tuple[str, float]]:
    severity_mapping = {Severity.minimal: 0.5, Severity.significant: 1.0}
    return [(token.text, severity_mapping.get(token.severity, 0.0)) for token in token_list]
import csv
from tqdm import tqdm

csv_file = "testset_prediction_text.csv"
reader = csv.DictReader(open(csv_file))
result = []
for line_dict in tqdm(reader):
    post_text = line_dict["Text"]
    all_tokens = tokenize_post_text(post_text)
    all_tokens_as_zero = [(token.text, 0.0) for token in all_tokens]
    line_result = {}
    for attribute_name, attribute_definition in ATTRIBUTE_DEFINITIONS.items():
        if int(float(line_dict[attribute_name])) == 0:
            result_tuple = (0, all_tokens_as_zero)
            line_result[attribute_name] = str(result_tuple)
        else:
            result_tuple = (1, analyse_post_for_attribute(post_text, attribute_name, attribute_definition))
            line_result[attribute_name] = str(result_tuple)
    result.append(line_result)

import pandas as pd

# convert to dataframe and then save as csv
df = pd.DataFrame(result)
df.to_csv("testset_prediction_text_result.csv", index=False)