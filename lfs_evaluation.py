import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from liap_core import get_model_predictions, apply_liap

import torch
import numpy as np
from liap_core import get_model_predictions

def calculate_lfs_perturbation(model, tokenizer, sentence, entity_position, pruned_attention, id2label, num_perturbations=10):
    """Calculate LFS based on softmax probability drops."""
    
    # Get original predictions
    original_logits, word_ids = get_model_predictions(model, tokenizer, sentence, return_logits=True)
    
    if entity_position >= len(original_logits):
        print(f"DEBUG: Entity position {entity_position} out of range")
        return None, "Entity position out of range"
    
    original_logits = original_logits[entity_position]
    original_probs = torch.softmax(original_logits, dim=-1)
    original_pred_id = original_probs.argmax().item()
    
    if id2label[original_pred_id] == "O":
        print(f"DEBUG: Position {entity_position} is not an entity (O tag)")
        return None, "Not an entity"
    
    tokens = sentence.split()
    
    # Token importance vector
    token_importance = pruned_attention.sum(dim=1)
    
    if len(token_importance) > len(tokens):
        token_importance = token_importance[:len(tokens)]
    elif len(token_importance) < len(tokens):
        token_importance = torch.nn.functional.pad(token_importance, (0, len(tokens) - len(token_importance)))

    if entity_position < len(token_importance):
        token_importance[entity_position] = 0
    
    # Top and bottom tokens by importance
    important_indices = token_importance.argsort(descending=True)[:num_perturbations]
    important_indices = [i.item() for i in important_indices if i < len(tokens)]
    
    unimportant_indices = token_importance.argsort(descending=False)[:num_perturbations]
    unimportant_indices = [i.item() for i in unimportant_indices if i < len(tokens) and i != entity_position]
    
    important_drop = 0.0
    unimportant_drop = 0.0
    
    # Perturb important tokens
    for idx in important_indices:
        perturbed_tokens = tokens.copy()
        perturbed_tokens[idx] = "[MASK]"
        perturbed_sentence = " ".join(perturbed_tokens)
        
        perturbed_logits, _ = get_model_predictions(model, tokenizer, perturbed_sentence, return_logits=True)
        if entity_position < len(perturbed_logits):
            perturbed_probs = torch.softmax(perturbed_logits[entity_position], dim=-1)
            delta = original_probs[original_pred_id] - perturbed_probs[original_pred_id]
            important_drop += max(delta.item(), 0.0)
    
    # Perturb unimportant tokens
    for idx in unimportant_indices:
        perturbed_tokens = tokens.copy()
        perturbed_tokens[idx] = "[MASK]"
        perturbed_sentence = " ".join(perturbed_tokens)
        
        perturbed_logits, _ = get_model_predictions(model, tokenizer, perturbed_sentence, return_logits=True)
        if entity_position < len(perturbed_logits):
            perturbed_probs = torch.softmax(perturbed_logits[entity_position], dim=-1)
            delta = original_probs[original_pred_id] - perturbed_probs[original_pred_id]
            unimportant_drop += max(delta.item(), 0.0)
    
    # Calculate LFS
    if important_drop == 0.0 and unimportant_drop == 0.0:
        lfs = 0.5  # Neutral
    else:
        lfs = important_drop / (important_drop + unimportant_drop)
    
    return lfs, {
        "original_entity": id2label[original_pred_id],
        "important_total_drop": important_drop,
        "unimportant_total_drop": unimportant_drop,
        "important_indices": important_indices,
        "unimportant_indices": unimportant_indices
    }
