import torch
import spacy
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

# Load spaCy and AllenNLP SRL model only once
nlp = spacy.load("en_core_web_sm")
srl_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)

def get_model_predictions(model, tokenizer, sentence, return_attention=False, return_logits=False):
    # Tokenize with offset_mapping but drop it before model input
    encoded = tokenizer(sentence, return_tensors="pt", truncation=True, return_offsets_mapping=True)
    
    inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"]

    # Safe model call
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    word_ids = encoded.word_ids(batch_index=0)

    logits = logits[0]
    predictions = logits.argmax(dim=-1)
    real_word_preds = []
    real_word_logits = []
    seen = set()
    
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id in seen:
            continue
        seen.add(word_id)
        real_word_preds.append(predictions[idx].item())
        real_word_logits.append(logits[idx])

    if return_logits:
        return real_word_logits, word_ids
    elif return_attention:
        attentions = outputs.attentions
        return real_word_preds, attentions, word_ids
    else:
        return real_word_preds, word_ids

def syntactic_dependency_pruning(sentence, attention_map, entity_position):
    doc = nlp(sentence)
    pruning_mask = torch.zeros_like(attention_map)
    entity_token = None
    for i, token in enumerate(doc):
        if i == entity_position:
            entity_token = token
            break
    if entity_token is None:
        return attention_map
    related_tokens = set()
    related_tokens.add(entity_token.i)
    related_tokens.add(entity_token.head.i)
    for child in entity_token.children:
        related_tokens.add(child.i)
    for token in doc:
        if token.head.i in related_tokens or token.i in related_tokens:
            related_tokens.add(token.i)
    for i in range(len(doc)):
        for j in range(len(doc)):
            if i in related_tokens or j in related_tokens:
                if i < pruning_mask.shape[0] and j < pruning_mask.shape[1]:
                    pruning_mask[i, j] = 1.0
    pruned_attention = attention_map * pruning_mask
    return pruned_attention

def srl_pruning(sentence, attention_map, entity_position):
    """
    Prune attention based on semantic roles using AllenNLP's SRL.
    Only keeps attention to tokens that are in the same semantic frame as the entity.
    """
    srl_result = srl_predictor.predict(sentence=sentence)
    tokens = srl_result["words"]
    # Find which SRL frame contains the entity_position
    relevant_indices = set()
    for verb in srl_result["verbs"]:
        tags = verb["tags"]
        for idx, tag in enumerate(tags):
            if idx == entity_position and tag != "O":
                # This entity is part of this SRL frame
                # Keep all tokens with non-"O" tag in this frame
                for j, t in enumerate(tags):
                    if t != "O":
                        relevant_indices.add(j)
    if not relevant_indices:
        # If no SRL frame found, don't prune
        return attention_map
    pruning_mask = torch.zeros_like(attention_map)
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if i in relevant_indices or j in relevant_indices:
                if i < pruning_mask.shape[0] and j < pruning_mask.shape[1]:
                    pruning_mask[i, j] = 1.0
    pruned_attention = attention_map * pruning_mask
    return pruned_attention

def apply_liap(
    model, tokenizer, sentence, entity_position, layer=-1, head=0,
    use_syntactic=True, use_semantic=True
):
    _, attentions, word_ids = get_model_predictions(model, tokenizer, sentence, return_attention=True)
    attention_map = attentions[layer][0, head].cpu()
    pruned_attention = attention_map.clone()
    if use_syntactic:
        pruned_attention = syntactic_dependency_pruning(sentence, pruned_attention, entity_position)
    if use_semantic:
        pruned_attention = srl_pruning(sentence, pruned_attention, entity_position)
    return attention_map, pruned_attention, word_ids

def visualize_attention(sentence, attention_map, pruned_attention, save_path=None):
    """Visualize original and pruned attention maps."""
    tokens = sentence.split()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original attention
    sns.heatmap(attention_map.numpy(), xticklabels=tokens, yticklabels=tokens, ax=ax1, cmap="YlOrRd")
    ax1.set_title("Original Attention")
    
    # Pruned attention
    sns.heatmap(pruned_attention.numpy(), xticklabels=tokens, yticklabels=tokens, ax=ax2, cmap="YlOrRd")
    ax2.set_title("LIAP-Pruned Attention")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
