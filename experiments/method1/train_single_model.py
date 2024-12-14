
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


model_name = 'microsoft/deberta-v3-large'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token #= tokenizer.special_tokens_map['pad_token']
tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer, classes=classes), batched=False)
tokenized_ds = tokenized_ds.with_format('torch')

labels = tokenized_ds['train']['labels']
label_weights = 1 / labels.mean(dim=0, dtype=torch.float32)

tokenized_ds = tokenized_ds.shuffle()

# qunatization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

# lora config
lora_config = LoraConfig(
    r = 8, # the dimension of the low-rank matrices
    lora_alpha = 16, # scaling factor for LoRA activations vs pre-trained weight activations
    # target_modules="all-linear",
    # target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    # target_modules = ['query_proj', 'value_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="cuda:0",
    quantization_config=quantization_config,
    num_labels=len(classes),
    problem_type="multi_label_classification",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id

# define training args
training_args = TrainingArguments(
    output_dir = 'multilabel_classification',
    logging_dir = 'multilabel_classification/logs',
    learning_rate = 1e-3,
    per_device_train_batch_size = 8, # tested with 16gb gpu ram
    per_device_eval_batch_size = 8,
    num_train_epochs = 5,
    eval_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True
)

trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_ds['train'],
    eval_dataset = tokenized_ds['validation'],
    tokenizer = tokenizer,
    data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
    compute_metrics = functools.partial(compute_metrics, id2class=id2class, classes=classes),
    label_weights = torch.tensor(label_weights, device=model.device)
)

trainer.train()

# save model
peft_model_id = 'multilabel_deberta_v3_large_peft'
model = model.merge_and_unload()
model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

repository_id = 'anismk/' + peft_model_id
model.push_to_hub(repository_id)
tokenizer.push_to_hub(repository_id)

results_dir = Path("../results/deberta-v3-large")
if not results_dir.exists():
    results_dir.mkdir(parents=True)
    
val_metrics = trainer.evaluate(tokenized_ds['validation'], metric_key_prefix="validation")
print(f"{val_metrics=}")
pd.DataFrame(val_metrics, index=[0]).to_csv(results_dir / "validation.csv")

test_metrics = trainer.evaluate(tokenized_ds['test'], metric_key_prefix="test")
print(f"{test_metrics=}")
pd.DataFrame(test_metrics, index=[0]).to_csv(results_dir / "test.csv")