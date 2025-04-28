import os
os.environ["TORCH_DIST_DISABLE"] = "1"
from visualizations import enhanced_attention_visualization
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import our modules
from liap_core import apply_liap, visualize_attention
from lfs_evaluation import calculate_lfs_perturbation
from lime_baseline import explain_with_lime

# Create output directories
os.makedirs("results", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)
os.makedirs("results/lime", exist_ok=True)

# Load model and tokenizer
model_path = "./best_ner_model"
model = AutoModelForTokenClassification.from_pretrained(
    model_path, 
    # attn_implementation="eager", 
    output_attentions=True, 
    # use_safetensors=True, 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load dataset
dataset = load_dataset("conll2003")
test_data = dataset["test"]

# Get label mappings
label_list = test_data.features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in id2label.items()}

# Select examples for evaluation
def select_evaluation_examples(dataset, num_examples=40):
    """Select diverse examples for evaluation with balanced entity types."""
    selected_examples = []
    entity_counts = {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}
    target_per_type = num_examples // 4  # 10 examples per entity type
    
    # First pass: try to get balanced examples
    for i, example in enumerate(dataset):
        tokens = example["tokens"]
        
        # Skip very long sentences and very short sentences
        if len(tokens) > 40 or len(tokens) < 8:
            continue
            
        ner_tags = example["ner_tags"]
        
        for j, (token, tag) in enumerate(zip(tokens, ner_tags)):
            tag_name = id2label[tag]
            
            # Check if it's a B- tag (beginning of entity)
            if tag_name.startswith("B-"):
                entity_type = tag_name[2:]
                
                # If we need more of this entity type
                if entity_counts[entity_type] < target_per_type:
                    entity_counts[entity_type] += 1
                    
                    # Create a sentence from tokens
                    sentence = " ".join(tokens)
                    
                    selected_examples.append({
                        "sentence": sentence,
                        "entity_position": j,
                        "entity_type": entity_type,
                        "entity_token": tokens[j]
                    })
                    break
    
    # Second pass: if we couldn't find enough examples in test set, try validation set
    if sum(entity_counts.values()) < num_examples * 0.75:  # If we couldn't find at least 75% of requested examples
        print(f"Test set provided only {sum(entity_counts.values())} examples. Trying validation set...")
        validation_data = dataset["validation"]
        
        for i, example in enumerate(validation_data):
            if sum(entity_counts.values()) >= num_examples:
                break
                
            tokens = example["tokens"]
            
            # Skip very long sentences
            if len(tokens) > 40:
                continue
                
            ner_tags = example["ner_tags"]
            
            for j, (token, tag) in enumerate(zip(tokens, ner_tags)):
                tag_name = id2label[tag]
                
                if tag_name.startswith("B-"):
                    entity_type = tag_name[2:]
                    
                    if entity_counts[entity_type] < target_per_type:
                        entity_counts[entity_type] += 1
                        sentence = " ".join(tokens)
                        
                        selected_examples.append({
                            "sentence": sentence,
                            "entity_position": j,
                            "entity_type": entity_type,
                            "entity_token": tokens[j]
                        })
                        break
    
    print(f"Selected examples by entity type: {entity_counts}")
    return selected_examples

# Select examples
evaluation_examples = select_evaluation_examples(test_data)

results = []

# Track which pruning strategies work best
strategy_performance = {
    "syntactic_only": [],
    "semantic_only": [],
    "both": [],
    "layer_wise": [],
    "differentiable": []
}

for idx, example in enumerate(tqdm(evaluation_examples, desc="Evaluating examples")):
    sentence = example["sentence"]
    entity_position = example["entity_position"]
    entity_type = example["entity_type"]
    entity_token = example["entity_token"]
    
    # Debug timing
    start_time = time.time()
    
    # Try different pruning strategies to find the best one
    strategies = [
        {"name": "syntactic_only", "use_syntactic": True, "use_semantic": False},
        {"name": "semantic_only", "use_syntactic": False, "use_semantic": True},
        {"name": "both", "use_syntactic": True, "use_semantic": True},
        {"name": "layer_wise", "use_syntactic": True, "use_semantic": True, "use_layer_wise": True},
        {"name": "differentiable", "use_syntactic": True, "use_semantic": True, "use_differentiable": True}
    ]
    
    best_lfs = -1
    best_strategy = None
    best_attention = None
    best_pruned = None
    
    # Try each strategy
    for strategy in strategies:
        # Apply LIAP with this strategy
        original_attention, pruned_attention, word_ids = apply_liap(
            model, tokenizer, sentence, entity_position,
            layer=-2, head=0,
            use_syntactic=True,   # Enable syntactic dependency pruning
            use_semantic=True     # Enable SRL pruning
        )
        
        # Calculate LFS for this strategy
        lfs, lfs_details = calculate_lfs_perturbation(
            model, tokenizer, sentence, entity_position,
            pruned_attention, id2label
        )
        
        # Skip if LFS couldn't be calculated
        if lfs is None:
            continue
            
        # Track strategy performance
        strategy_performance[strategy["name"]].append(lfs)
        
        # Update best strategy if this one is better
        if lfs > best_lfs:
            best_lfs = lfs
            best_strategy = strategy["name"]
            best_attention = original_attention
            best_pruned = pruned_attention
    
    elapsed = time.time() - start_time
    print(f"Example {idx}: LIAP on sentence (length={len(sentence.split())} tokens) took {elapsed:.2f} seconds")
    
    # If no strategy worked, use the default
    if best_strategy is None:
        original_attention, pruned_attention, word_ids = apply_liap(
            model, tokenizer, sentence, entity_position,
            layer=-2, head=0,
            use_syntactic=True,   # Enable syntactic dependency pruning
            use_semantic=True     # Enable SRL pruning
        )
        best_attention = original_attention
        best_pruned = pruned_attention
        best_strategy = "default"
        
        # Calculate LFS
        best_lfs, lfs_details = calculate_lfs_perturbation(
            model, tokenizer, sentence, entity_position,
            pruned_attention, id2label
        )
    else:
        lfs_details = None  # We don't have details for the best strategy

    print(f"LFS: {lfs}")
    
    # Get LIME explanation
    lime_explanation, lime_details = explain_with_lime(
        model, tokenizer, sentence, entity_position, id2label
    )
    
    # Save enhanced visualization
    viz_path = f"results/visualizations/example_{len(results)}.png"
    enhanced_attention_visualization(
        sentence, best_attention, best_pruned, 
        entity_position, entity_type, 
        lime_explanation, 
        save_path=viz_path
    )
    
    # Save LIME HTML
    if isinstance(lime_details, dict) and "html" in lime_details:
        with open(f"results/lime/example_{len(results)}.html", "w") as f:
            f.write(lime_details["html"])
    
    # Store results
    result = {
        "example_id": len(results),
        "sentence": sentence,
        "entity_position": entity_position,
        "entity_token": entity_token,
        "entity_type": entity_type,
        "lfs": best_lfs,
        "best_strategy": best_strategy,
        "lfs_details": lfs_details,
        "lime_explanation": lime_explanation,
        "lime_details": lime_details if isinstance(lime_details, dict) else None,
        "visualization_path": viz_path
    }
    
    results.append(result)


# Save results
with open("results/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Calculate overall statistics
lfs_scores = [r["lfs"] for r in results if r["lfs"] is not None]
avg_lfs = np.mean(lfs_scores) if lfs_scores else 0

print(f"Evaluation completed for {len(results)} examples")
print(f"Average LFS: {avg_lfs:.4f}")

# Generate summary visualizations
plt.figure(figsize=(10, 6))
sns.histplot(lfs_scores, bins=10)
plt.title("Distribution of Linguistic Faithfulness Scores")
plt.xlabel("LFS")
plt.ylabel("Count")
plt.savefig("results/lfs_distribution.png")

# Create summary by entity type
entity_lfs = {}
for r in results:
    if r["lfs"] is not None:
        if r["entity_type"] not in entity_lfs:
            entity_lfs[r["entity_type"]] = []
        entity_lfs[r["entity_type"]].append(r["lfs"])

plt.figure(figsize=(10, 6))
data = []
labels = []
for entity_type, scores in entity_lfs.items():
    if scores:  # Only include non-empty score lists
        data.append(scores)
        labels.append(f"{entity_type} (n={len(scores)})")

if data:  # Only create boxplot if there's data to plot
    plt.boxplot(data, labels=labels)
    plt.title("LFS by Entity Type")
    plt.ylabel("Linguistic Faithfulness Score")
    plt.savefig("results/lfs_by_entity_type.png")
else:
    print("No valid LFS scores to plot by entity type")

print("Analysis completed. Results saved to the 'results' directory.")
