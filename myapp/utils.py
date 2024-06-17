from .models import ClassificationResult

def classify_patent(patent):
    # Logique de classification du brevet (à implémenter)
    predicted_category = "Example Category"  # Remplacez par la sortie réelle de votre modèle ML
    explanation = "Example explanation"  # Remplacez par l'explication générée par votre modèle ML

    classification_result = ClassificationResult.objects.create(
        patent=patent,
        predicted_category=predicted_category,
        explanation=explanation
    )
    return classification_result