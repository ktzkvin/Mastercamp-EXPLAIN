from django.shortcuts import render, get_object_or_404, redirect
from .models import Patent, ClassificationResult
from .forms import PatentForm, FeedbackForm
from .utils import classify_patent
from django.http import HttpResponse
import joblib
import numpy as np
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    # Charger les embeddings et les autres données nécessaires
    sample_df = joblib.load('sample_df_with_embeddings.pkl')
    log_reg_model = joblib.load('log_reg_model_multilabel.pkl')
    scaler = joblib.load('scaler.pkl')

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

    # Prendre un échantillon de données de test
    X_test_sample = np.array(X_test_scaled[sample_index].reshape(1, -1))
    explanation = explainer.explain_instance(X_test_sample[0], log_reg_model.predict_proba, num_features=10)

    return explanation

def explain_view(request, sample_index):
    explanation = generate_explanation(sample_index)
    explanation_html = explanation.as_html()
    return render(request, 'myapp/explain.html', {'explanation_html': explanation_html})
