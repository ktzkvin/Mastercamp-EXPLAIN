import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def classify_patent(patent):
    tokenizer = AutoTokenizer.from_pretrained('anferico/bert-for-patents')
    model = AutoModelForSequenceClassification.from_pretrained('anferico/bert-for-patents')
    model.eval()

    text = patent.description
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_category = logits.argmax().item()

    # Utiliser SHAP pour l'explication
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer(text)

    explanation = shap_values.data

    classification_result = ClassificationResult(
        patent=patent,
        predicted_category=predicted_category,
        explanation=explanation
    )
    classification_result.save()
    return classification_result