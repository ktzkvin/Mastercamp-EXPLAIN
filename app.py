from flask import Flask, render_template
import joblib
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Charger les données et les modèles une seule fois au démarrage de l'application
sample_df = joblib.load('sample_df_with_updated_embeddings.pkl')
log_reg_model = joblib.load('log_reg_model_balanced.pkl')
scaler = joblib.load('scaler_balanced.pkl')

# Fonction pour générer une explication LIME
def generate_lime_explanation(index):
    selected_sample = sample_df.iloc[index]
    application_number = selected_sample["Numéro d'application"]
    description = selected_sample["infos_essentielles"]

    X = np.array(selected_sample["embeddings_bert"]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    embeddings = np.vstack(sample_df["embeddings_bert"].dropna().values)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        embeddings,
        feature_names=["Embedding_" + str(i) for i in range(X_scaled.shape[1])],
        class_names=log_reg_model.classes_,
        discretize_continuous=True,
    )

    exp = explainer.explain_instance(X_scaled[0], log_reg_model.predict_proba, num_features=10)
    exp_map = exp.as_map()

    print(f"Explanation map: {exp_map}")

    feature_weights = []
    for feature, weight in exp_map[1]:
        try:
            feature_name = explainer.feature_names[feature]
            feature_weights.append((feature_name, weight))
        except IndexError:
            continue

    print(f"Feature weights: {feature_weights}")

    important_words = extract_important_words(description, feature_weights)
    print(f"Important words: {important_words}")

    predict_proba = exp.predict_proba
    if predict_proba.ndim == 2:
        predict_proba = predict_proba[0]

    explanation = {
        "application_number": application_number,
        "predict_proba": [(log_reg_model.classes_[i], prob) for i, prob in enumerate(predict_proba)],
        "feature_weights": feature_weights,
        "predicted_class": log_reg_model.classes_[np.argmax(predict_proba)],
        "predicted_prob": max(predict_proba),
        "important_words": important_words,
    }

    return explanation

# Fonction pour extraire les mots importants
def extract_important_words(description, feature_weights):
    # Création d'un dictionnaire de mots avec leur poids
    word_weights_dict = {int(feature.split('_')[1]): weight for feature, weight in feature_weights}

    word_weights = []
    for segment in description:
        words = segment.split()
        segment_weights = []
        for i, word in enumerate(words):
            # Assurez-vous que l'index de l'embedding est correct et mappé au bon mot
            weight = word_weights_dict.get(i, 0)  # Utilisation de l'indice du mot
            segment_weights.append((word, weight))
        word_weights.append(segment_weights)
    return word_weights


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/import_patent")
def import_patent():
    return render_template("import_patent.html")

@app.route("/classification_result")
def classification_result():
    example_classification_result = {
        "patent": {"application_number": "123456789"},
        "predicted_category": "Category A",
        "explanation": "This is the explanation for the classification result.",
    }
    return render_template("classification_result.html", classification_result=example_classification_result)

@app.route("/explain/<int:index>")
def explain_view(index):
    explanation = generate_lime_explanation(index)
    return render_template("explain.html", index=index, explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)
