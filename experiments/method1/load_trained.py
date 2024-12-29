
import functools
import torch
import torch.nn.functional as F

from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import precision_recall_fscore_support

from datasets import load_dataset

from pathlib import Path
import pandas as pd

from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
login(os.getenv("HF_TOKEN"))


def tokenize_examples(examples, tokenizer, classes):
    text = f"Issue: {examples['issue']}.\nAnswer: {examples['post_text']}"
    labels = [examples[label] for label in classes]
    tokenized_inputs = tokenizer(text, truncation=True, max_length=700, padding=True)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


# define custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels']).type(torch.float)
    return d


# define which metrics to compute for evaluation
def compute_metrics(p, id2class, classes):
    predictions, labels = p
    predictions_binary = predictions > 0
    
    metrics = {}
    for j, dim in enumerate(classes):
        scores = precision_recall_fscore_support(
            [x[j] for x in labels], [x[j] for x in predictions_binary], average="macro"
        )
        metrics["Macro-F1 " + dim] = scores[2]
    return metrics


# create custom trainer class to be able to pass label weights and calculate mutilabel loss
class CustomTrainer(Trainer):

    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights
    
    def compute_loss(self, model, inputs, num_items_in_batch=1000, return_outputs=False):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss

    
ds = load_dataset('timonziegenbein/appropriateness-corpus')

classes = [
    'Toxic Emotions',
    'Missing Commitment',
    'Missing Intelligibility',
    'Other Reasons',
    'Inappropriateness',
    'Excessive Intensity',
    'Emotional Deception',
    'Missing Seriousness',
    'Missing Openness',
    'Unclear Meaning',
    'Missing Relevance',
    'Confusing Reasoning',
    'Detrimental Orthography',
    'Reason Unclassified'
]
class2id = {class_:id for id, class_ in enumerate(classes)}
id2class = {id:class_ for class_, id in class2id.items()}


model_name = './multilabel_deberta_v3_large_peft'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token #= tokenizer.special_tokens_map['pad_token']
tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer, classes=classes), batched=False)
tokenized_ds = tokenized_ds.with_format('torch')

labels = tokenized_ds['train']['labels']
label_weights = 1 / labels.mean(dim=0, dtype=torch.float32)

tokenized_ds = tokenized_ds.shuffle()


# lora config
lora_config = LoraConfig(
    r = 8, # the dimension of the low-rank matrices
    lora_alpha = 16, # scaling factor for LoRA activations vs pre-trained weight activations
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

# load model
# qunatization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="cuda:0",
    quantization_config=quantization_config,
    num_labels=len(classes),
    problem_type="multi_label_classification",
)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

# Create trainer with custom collate function and metrics
trainer = CustomTrainer(
    model=model,
    label_weights=label_weights.to('cuda'),
    args=TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,
        remove_unused_columns=False,
    ),
    eval_dataset=tokenized_ds['test'],
    compute_metrics=lambda p: compute_metrics(p, id2class, classes),
    data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
)

# Run evaluation
eval_results = trainer.evaluate()

# Print results
for metric, value in eval_results.items():
    print(f"{metric}: {value:.4f}")