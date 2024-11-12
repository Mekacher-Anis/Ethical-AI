# Ethical-AI (Modeling Appropriate Language in Argumentation)

This repo contains a classifier for appropriate language in argumentation,
with model output explanation using word importance.

## Dataset
We use the following dataset
https://huggingface.co/datasets/timonziegenbein/appropriateness-corpus

## Training

### Method 1
Fine tuning a pretrained BERT model for classification.
For that we use [LoRA](https://arxiv.org/abs/2106.09685) with a 4-bit quantized model using huggingface transformer with PEFT, bitsandbytes.

### Method 2
tbd.

## Explainers

### For method 1


## Collaborators
- [ErikVogelLUH](https://github.com/ErikVogelLUH)