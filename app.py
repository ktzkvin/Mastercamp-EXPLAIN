from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/import_patent')
def import_patent():
    # Logique pour l'importation d'un brevet
    return render_template('import_patent.html')

@app.route('/classification_result')
def classification_result():
    # Exemples de données de classification
    example_classification_result = {
        'patent': {
            'application_number': '123456789'
        },
        'predicted_category': 'Category A',
        'explanation': 'This is the explanation for the classification result.'
    }

    # Passez example_classification_result au template
    return render_template('classification_result.html', classification_result=example_classification_result)

@app.route('/explain/<int:index>')
def explain_view(index):
    # Exemple de données d'explication LIME pour un échantillon
    example_explanation = {
        'predict_proba': [('Class 1', 0.7), ('Class 2', 0.3)],
        'feature_weights': [('Feature 1', 0.5), ('Feature 2', -0.3)],
        'predicted_class': 'Class 1',
        'predicted_prob': 0.7
    }

    # Passez example_explanation et index au template
    return render_template('explain.html', index=index, explanation=example_explanation)

if __name__ == "__main__":
    app.run(debug=True)
