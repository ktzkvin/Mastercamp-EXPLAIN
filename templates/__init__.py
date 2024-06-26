from flask import Flask

# Créer une instance de l'application Flask
app = Flask(__name__, template_folder='templates')

# Charger une configuration spécifique (optionnel)
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Importer des vues ou des modèles (exemple)
from myapp import views