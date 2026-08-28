# Importing the libraries
import pandas as pd
import numpy as np
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR # Understand this
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoConfig, TrainingArguments, Trainer, TrainerCallback, DataCollatorWithPadding,
    AutoModelForSequenceClassification, EvalPrediction, get_scheduler, trainer_callback
    )
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
import os
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score
from datasets import Dataset, load_from_disk
import gc
import transformers
import datasets
from torch.nn import BCEWithLogitsLoss

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "Finetuning" / "data"
DEV_DATA_PATH = DATA_DIR / "processed" / "dev.csv"
TEST_DATA_PATH = DATA_DIR / "processed" / "test.csv"
TRAIN_DATA_PATH = DATA_DIR / "processed" / "train.csv"

OUTPUT_DIR = PROJECT_ROOT / "models"

TEXT_COL = "comment_text"
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

MODELS_TO_FINETUNE = [
    "GroNLP/hateBERT",
    "roberta-base",
    "distilbert-base-uncased"
]

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(DEVICE)

# Importing the dataset
train_df = pd.read_csv(TRAIN_DATA_PATH)
dev_df = pd.read_csv(DEV_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

train_dataset = Dataset.from_pandas(train_df[[TEXT_COL] + LABEL_COLS])
dev_dataset = Dataset.from_pandas(dev_df[[TEXT_COL] + LABEL_COLS])
test_dataset = Dataset.from_pandas(test_df[[TEXT_COL] + LABEL_COLS])


def tokenize_function(tokenizer, max_length):
    def inner(batch):
        return tokenizer(
            batch[TEXT_COL],
            max_length=max_length,
            truncation=True
        )
    return inner # returns the function not the result.


def format_labels(example):
    return {
        **example,
        "labels": [
            float(example["toxic"]),
            float(example["severe_toxic"]),
            float(example["obscene"]),
            float(example["threat"]),
            float(example["insult"]),
            float(example["identity_hate"]),
        ]
    }


for model_name in MODELS_TO_FINETUNE:
    config = AutoConfig.from_pretrained(model_name)
    print(model_name, config.vocab_size)


pos_weight = torch.tensor([3.08, 10.63, 4.18, 17.97, 4.35, 10.24])

for model_name in MODELS_TO_FINETUNE:
    torch.cuda.empty_cache()

    safe_model_name = model_name.replace("/", "_")
    model_ckpt_dir = OUTPUT_DIR / safe_model_name
    os.makedirs(model_ckpt_dir,
                exist_ok=True)
                
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=6,
        problem_type="multi_label_classification"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=config,
        ignore_mismatched_sizes=True,
        )

    print('#1: Models and tokenizers created')

    max_length = int(config.max_position_embeddings) - 2

    tokenized_train_dataset = train_dataset.map(
        tokenize_function(tokenizer=tokenizer, max_length=max_length),
        batched=True,
        batch_size=1024,
        remove_columns=TEXT_COL
    )

    tokenized_dev_dataset = dev_dataset.map(
        tokenize_function(tokenizer=tokenizer, max_length=max_length),
        batched=True,
        batch_size=1024,
        remove_columns=TEXT_COL
    )

    tokenized_train_dataset = tokenized_train_dataset.map(
        format_labels,
        remove_columns=LABEL_COLS
    )

    tokenized_train_dataset = tokenized_train_dataset.cast_column(
        "labels",
        datasets.Sequence(datasets.Value("float32"))
    )

    tokenized_dev_dataset = tokenized_dev_dataset.map(
        format_labels,
        remove_columns=LABEL_COLS
    )

    tokenized_dev_dataset = tokenized_dev_dataset.cast_column(
        "labels",
        datasets.Sequence(datasets.Value("float32"))
    )

    tokenized_train_dataset.save_to_disk(DATA_DIR / "tokenized" / "Train")
    tokenized_dev_dataset.save_to_disk(DATA_DIR / "tokenized" / "Dev")

    print("2. Train & Dev datasets tokenized and save with arrow format.")

    train_dataset_disk = load_from_disk(
        f"/Users/mouldi/Downloads/results/Train_{safe_model_name}"
    )

    dev_dataset_disk = load_from_disk(
        f"/Users/mouldi/Downloads/results/Dev_{safe_model_name}"
    )

    # NOTE Add "token_type_ids" to columns if using HateBERT.
    train_dataset_disk.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    dev_dataset_disk.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    print(f"[{model_name}] Total train samples: {tokenized_train_dataset.num_rows}")
    print(f"[{model_name}] Total dev samples: {tokenized_dev_dataset.num_rows}")

    print('#3: Datasets imported')

    num_epochs                  = 4
    train_batch_size            = 32 
    gradient_accumulation_steps = 2

    total = train_dataset_disk.num_rows

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = train_batch_size * num_gpus * gradient_accumulation_steps
    print(f'---Effective batch size: {effective_batch_size}')

    steps_per_epoch = math.ceil(total / effective_batch_size)
    num_training_steps = steps_per_epoch * num_epochs
    print(f'---Total number of training steps: {num_training_steps}')

    warmup_ratio = 0.05
    decay_ratio  = 0.10
    num_warmup_steps   = int(warmup_ratio * num_training_steps)
    num_decay_steps    = int(decay_ratio  * num_training_steps)
    num_constant_steps = num_training_steps - num_warmup_steps - num_decay_steps

    print(f"Number of warmup steps: {num_warmup_steps} \n    Number of decay steps: {num_decay_steps}.")

    def warmup_constant_final_decay(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_constant_steps:
            return 1.0
        decay_step = current_step - num_warmup_steps - num_constant_steps
        return max(0.0, 1.0 - float(decay_step) / float(max(1, num_decay_steps)))

    optimizer    = AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01, fused=True
        )
    
    lr_scheduler = LambdaLR(
        optimizer, lr_lambda=warmup_constant_final_decay
        )

    print("#3: Optimizer object created")

    output_dir = model_ckpt_dir
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=250,
        load_best_model_at_end=True, # Overwrites the initial model object
        metric_for_best_model="f1",
        greater_is_better=True,
        gradient_accumulation_steps=gradient_accumulation_steps,# How does this work and benefits
        report_to="none", # Try "tensorboard" next.
        fp16=True,
        remove_unused_columns=True, # Understand this
        group_by_length=True,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False
    )

    class LearningRateLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return

            step = state.global_step
            parts = [f"step={step}"]

            key_order = [
                "loss", "eval_loss",
                "eval_precision", "eval_recall", "eval_f1",
                "learning_rate",
            ]
            # Print known keys in a fixed order first
            for key in key_order:
                if key in logs:
                    val = logs[key]
                    if key == "learning_rate":
                        parts.append(f"{key}={val:.2e}")
                    else:
                        parts.append(f"{key}={val:.4f}")

            # Then print anything else Trainer adds (epoch, runtime, etc.)
            for key, val in logs.items():
                if key not in key_order:
                    if isinstance(val, float):
                        parts.append(f"{key}={val:.4f}")
                    else:
                        parts.append(f"{key}={val}")

            print("  |  ".join(parts), flush=True)

        def on_epoch_end(self, args, state, control, **kwargs):
            print(f"\n{'='*60}", flush=True)
            print(f"Epoch {state.epoch:.0f} complete", flush=True)
            print(f"{'='*60}\n", flush=True)

    def compute_metrics(eval_preds: EvalPrediction):
        logits = eval_preds.predictions
        labels = eval_preds.label_ids

        probs  = 1 / (1 + np.exp(-logits))
        preds  = (probs >= 0.5).astype("int32")
        labels = labels.astype("int32")

        print("Positive labels per class:    ", labels.sum(axis=0))
        print("Predicted positives per class:", preds.sum(axis=0))

        return {
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall":    recall_score(   labels, preds, average="macro", zero_division=0),
            "f1":        f1_score(       labels, preds, average="macro", zero_division=0),
        }
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Understand this
    
    class WeightedTrainer(Trainer):
        def __init__(self, pos_weight, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pos_weight = pos_weight  # store raw, don't move yet

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            loss_fn = BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device)
            )
            loss = loss_fn(logits, labels.float())
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_disk,
        eval_dataset=dev_dataset_disk,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[LearningRateLogger()],
        optimizers=(optimizer, lr_scheduler)
    )

    print("#5: Trainer created")

    trainer.train()

    # NOTE: Why do we save the tokenizer, and what are we actually saving, also review tokenizer.
    safe_model_name = model_name.replace("/", "_")
    trainer.save_model(f"/kaggle/working/best_model_{safe_model_name}")
    tokenizer.save_pretrained(f"/kaggle/working/tokenizer_{safe_model_name}")

    del model, tokenizer, optimizer, lr_scheduler, trainer
    del tokenized_train_dataset, tokenized_dev_dataset
    del train_dataset_disk, dev_dataset_disk
    torch.cuda.empty_cache()
    gc.collect()
