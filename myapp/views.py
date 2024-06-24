from django.shortcuts import render, get_object_or_404, redirect
from .models import Patent, ClassificationResult
from .forms import PatentForm, FeedbackForm
from .utils import classify_patent
import joblib
import numpy as np
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Charger les gros fichiers une seule fois au démarrage du serveur
#SAMPLE_DF = joblib.load('sample_df_with_embeddings.pkl')
#LOG_REG_MODEL = joblib.load('log_reg_model_multilabel.pkl')
#SCALER = joblib.load('scaler.pkl')

def import_patent(request):
    if request.method == 'POST':
        form = PatentForm(request.POST)
        if form.is_valid():
            patent = form.save()
            classification_result = classify_patent(patent)
            return redirect('classification_result', pk=classification_result.pk)
    else:
        form = PatentForm()
    return render(request, 'myapp/import_patent.html', {'form': form})

def classification_result(request, pk):
    classification_result = get_object_or_404(ClassificationResult, pk=pk)
    if request.method == 'POST':
        form = FeedbackForm(request.POST, instance=classification_result)
        if form.is_valid():
            form.save()
            return redirect('classification_result', pk=pk)
    else:
        form = FeedbackForm(instance=classification_result)
    return render(request, 'myapp/classification_result.html',
                  {'classification_result': classification_result, 'form': form})

def home(request):
    return render(request, 'myapp/home.html')

def binarize_labels(labels, label_to_index):
    binarized = np.zeros(len(label_to_index), dtype=int)
    for label in labels:
        if label in label_to_index:
            binarized[label_to_index[label]] = 1
    return binarized

def generate_explanation(sample_index):
    sample_df = SAMPLE_DF  # Utiliser la variable globale chargée au démarrage
    log_reg_model = LOG_REG_MODEL  # Utiliser la variable globale chargée au démarrage
    scaler = SCALER  # Utiliser la variable globale chargée au démarrage

    # Vérifier si la colonne 'labels' est présente, sinon la recréer
    if 'labels' not in sample_df.columns:
        def extract_first_letters(cpc_list):
            if isinstance(cpc_list, str):
                try:
                    cpc_list = eval(cpc_list)
                except Exception as e:
                    print(f"Error evaluating CPC list: {e}")
                    return []
            if isinstance(cpc_list, list) and len(cpc_list) > 0:
                return list(set(code[0] for code in cpc_list if len(code) > 0))
            return []
        sample_df['labels'] = sample_df['CPC'].apply(extract_first_letters)

    # Obtenir toutes les lettres uniques dans l'ensemble de données
    unique_labels = sorted(set(letter for sublist in sample_df['labels'] for letter in sublist))

    # Créer un dictionnaire pour mapper chaque lettre unique à une colonne
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # Préparer les données
    X = np.array(sample_df['embeddings'].tolist())
    y = np.array([binarize_labels(labels, label_to_index) for labels in sample_df['labels']])

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardiser les données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Créer un explainer LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled,
        feature_names=['Embedding_' + str(i) for i in range(X_train_scaled.shape[1])],
        class_names=unique_labels,
        discretize_continuous=True
    )

    # Prendre un échantillon de données de test pops
    X_test_sample = np.array(X_test_scaled[sample_index].reshape(1, -1))
    explanation = explainer.explain_instance(X_test_sample[0], log_reg_model.predict_proba, num_features=10)

    return explanation


def explain_view(request, index):
    explanation = generate_explanation(index)
    explanation_data = {
        'predict_proba': list(zip(explanation.class_names, explanation.predict_proba)),
        'feature_weights': explanation.as_list(),
        'predicted_class': explanation.class_names[np.argmax(explanation.predict_proba)],
        'predicted_prob': float(np.max(explanation.predict_proba))
    }
    return render(request, 'myapp/explain.html', {'explanation': explanation_data, 'index': index})
