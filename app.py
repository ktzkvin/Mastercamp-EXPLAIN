from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}



# Créer le dossier 'uploads' s'il n'existe pas
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

sample_df = joblib.load('sample_df_with_updated_embeddings.pkl')
log_reg_model = joblib.load('log_reg_model_balanced.pkl')
scaler = joblib.load('scaler_balanced.pkl')



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_lime_explanation(index):
    selected_sample = sample_df.iloc[index]
    application_number = selected_sample["Numéro d'application"]
    description = selected_sample['infos_essentielles']

    X = np.array(selected_sample['embeddings_bert']).reshape(1, -1)
    X_scaled = scaler.transform(X)

    embeddings = np.vstack(sample_df['embeddings_bert'].dropna().values)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        embeddings,
        feature_names=['Embedding_' + str(i) for i in range(X_scaled.shape[1])],
        class_names=log_reg_model.classes_,
        discretize_continuous=True
    )

    exp = explainer.explain_instance(X_scaled[0], log_reg_model.predict_proba, num_features=10)
    exp_map = exp.as_map()
    feature_weights = []
    for feature, weight in exp_map[1]:
        try:
            feature_name = exp.domain_mapper.map_exp_ids([feature])[0]
            feature_weights.append((feature_name, weight))
        except IndexError:
            continue

    important_words = extract_important_words(description, feature_weights)

    explanation = {
        'application_number': application_number,
        'description': description,
        'predict_proba': [(log_reg_model.classes_[i], prob) for i, prob in enumerate(exp.predict_proba[0])] if len(
            exp.predict_proba.shape) > 1 else [(log_reg_model.classes_[0], exp.predict_proba)],
        'feature_weights': feature_weights,
        'predicted_class': log_reg_model.classes_[np.argmax(exp.predict_proba[0])] if len(
            exp.predict_proba.shape) > 1 else log_reg_model.classes_[0],
        'predicted_prob': max(exp.predict_proba[0]) if len(exp.predict_proba.shape) > 1 else exp.predict_proba,
        'important_words': important_words
    }

    return explanation


def extract_important_words(description, feature_weights):
    word_weights = []
    for segment in description:
        words = segment.split()
        segment_weights = []
        for word in words:
            weight = sum(weight for feature, weight in feature_weights if feature in word)
            segment_weights.append((word, weight))
        word_weights.append(segment_weights)
    return word_weights


def truncate_text(text, max_length=100):
    if pd.isna(text):
        return ''
    if isinstance(text, list):
        text = ' '.join(text)
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text

@app.route('/')
def home():
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('search', '', type=str)
    reset = request.args.get('reset', None)

    if reset:
        search_query = ''

    if search_query:
        sample_df["Numéro d'application"] = sample_df["Numéro d'application"].astype(str).fillna('')
        df_filtered = sample_df[sample_df["Numéro d'application"].str.contains(search_query)]
    else:
        df_filtered = sample_df

    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    total_pages = (len(df_filtered) + per_page - 1) // per_page

    df_subset = df_filtered.iloc[start:end]
    df_records = df_subset.to_dict(orient='records')

    for idx, record in enumerate(df_records, start=start):
        record['index'] = idx

    for record in df_records:
        record['description'] = truncate_text(record.get('description', ''), max_length=100)
        record['infos_essentielles'] = truncate_text(record.get('infos_essentielles', ''), max_length=100)
        record['claim'] = truncate_text(record.get('claim', ''), max_length=100)

    return render_template('home.html', df_records=df_records, page=page, total_pages=total_pages,
                           search_query=search_query)

@app.route('/import_patent')
def import_patent():
    return render_template('import_patent.html')

@app.route('/upload_patent', methods=['POST'])
def upload_patent():
    global sample_df
    application_number = request.form['application_number']
    cpc = request.form['cpc']
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        with open(file_path, 'r') as f:
            infos_essentielles = f.read()

        if application_number not in sample_df["Numéro d'application"].values:
            new_record = pd.DataFrame([{
                "Numéro d'application": application_number,
                "CPC": cpc,
                "infos_essentielles": infos_essentielles,
                # Ajoutez ici toute autre colonne nécessaire
            }])
            sample_df = pd.concat([sample_df, new_record], ignore_index=True)
            sample_df.to_pickle('sample_df_with_updated_embeddings.pkl')  # Sauvegarder les nouvelles données

            flash('Brevet importé avec succès!', 'success')
        else:
            flash('Le numéro d\'application existe déjà.', 'danger')
    else:
        flash('Format de fichier non supporté.', 'danger')

    return redirect(url_for('import_patent'))



@app.route('/classification_result')
def classification_result():
    example_classification_result = {
        'patent': {'Numéro d\'application': '123456789'},
        'predicted_category': 'Category A',
        'explanation': 'This is the explanation for the classification result.'
    }
    return render_template('classification_result.html', classification_result=example_classification_result)


@app.route('/explain/<int:index>')
def explain_view(index):
    explanation = generate_lime_explanation(index)
    return render_template('explain.html', index=index, explanation=explanation)


if __name__ == "__main__":
    app.run(debug=True)