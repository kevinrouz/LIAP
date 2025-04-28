import numpy as np
from lime.lime_text import LimeTextExplainer
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def get_prediction_function(model, tokenizer, entity_position, id2label):
    """Create a prediction function for LIME."""
    def predict_proba(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        seq_len = inputs["input_ids"].shape[1]
        center_position = seq_len // 2
        
        probs = torch.nn.functional.softmax(logits[:, center_position, :], dim=-1)
        
        return probs.cpu().numpy() 

    return predict_proba

def explain_with_lime(model, tokenizer, sentence, entity_position, id2label, num_features=5, num_samples=100):
    """Generate LIME explanation for a specific entity prediction."""
    explainer = LimeTextExplainer(class_names=list(id2label.values()))

    predict_fn = get_prediction_function(model, tokenizer, entity_position, id2label)

    predictions = predict_fn([sentence])
    original_pred = np.argmax(predictions[0])

    if id2label[original_pred] == "O":
        return None, "Not an entity"

    exp = explainer.explain_instance(
        sentence,
        predict_fn,
        num_features=num_features,
        labels=[original_pred],
        num_samples=num_samples
    )

    explanation = exp.as_list(label=original_pred)
    html_explanation = exp.as_html()

    return explanation, {
        "original_entity": id2label[original_pred],
        "html": html_explanation
    }
