import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
import evaluate
from typing import Dict, List, Optional

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003")
label_list = dataset["train"].features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in id2label.items()}

# Initialize Tokenizer and Model using BERT base
model_checkpoint = "bert-base-cased"  # Using cased model since NER is case sensitive
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256, 
        return_tensors="pt"
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # Special tokens have word_id set to None
            if word_idx is None:
                label_ids.append(-100)
            # For the first token of a word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For subsequent subword tokens
            else:
                # Use the same label for subwords or -100 to mask
                # Using the same label is typically better for NER
                label_ids.append(label[word_idx])
                
            previous_word_idx = word_idx
            
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization to datasets
tokenized_datasets = dataset.map(
    tokenize_and_align_labels, 
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Set up evaluation metrics
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Initialize model with proper number of labels
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

# Define better training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    logging_steps=100,
    warmup_steps=500,
    fp16=True,  # Mixed precision training
)

# Create a data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on validation set
validation_results = trainer.evaluate()
print(f"Validation results: {validation_results}")

# Test set evaluation
test_results = trainer.evaluate(tokenized_datasets["test"])
print(f"Test results: {test_results}")

# Save the model
model.save_pretrained("./best_ner_model", safe_serialization=False)
tokenizer.save_pretrained("./best_ner_model")

print("Training completed and model saved to ./best_ner_model")