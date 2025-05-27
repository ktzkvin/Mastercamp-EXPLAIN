import streamlit as st
import pandas as pd
import torch
import numpy as np
import joblib
import os
import re

from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from transformers import (
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModel,
    PegasusForConditionalGeneration,
    PegasusTokenizer
)

import shap
import sentencepiece as spm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
MAX_LEN = 200

cpc_mapping = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'Y'
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.max_len = max_len
        self.text = df.text.astype(str)
        self.tokenizer = tokenizer
        self.targets = df[CPC_list].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index >= len(self.text):
            raise IndexError("Index out of bounds")
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text, truncation=True, add_special_tokens=True, 
            max_length=self.max_len, padding='max_length', 
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained('roberta-base')
        self.fc = torch.nn.Linear(768, 9)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.fc(features)
        return output

# Set page configuration
st.set_page_config(
    page_title="PatentSense",
    page_icon="./assets/logo_PatentSense_v1.png",
)

# Title of the Streamlit app
st.title("Welcome to PatentSense 👋")

# Load pre-trained models and other necessary files
@st.cache_resource
def load_model():
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    model_cpc = joblib.load(MODEL_PATH)
    tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
    return tokenizer_bert, model_bert, model_cpc, tokenizer_roberta

tokenizer_bert, model_bert, model_cpc, tokenizer_roberta = load_model()

# Function to remove HTML tags
def remove_html_tags(text):
    if isinstance(text, str):
        text = re.sub(r'(<.*?>)', r' \1 ', text)
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text
    return text

def prepare_text(text, tokenizer, max_len):
    inputs = tokenizer.encode_plus(
        text,
        truncation=True,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_token_type_ids=True
    )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(device)
    return ids, mask, token_type_ids

# Function to extract features using BERT
@st.cache_data
def extract_features(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to preprocess and predict
def preprocess_and_predict(text):
    # Predict CPC class
    predictions = predict(text, model_cpc, tokenizer_roberta, MAX_LEN)
    interpreted_predictions = interpret_predictions(predictions)

    indices_of_ones = [i for i, val in enumerate(interpreted_predictions[0]) if val == 1]
    cpc_letters = [cpc_mapping[index] for index in indices_of_ones]

    if not cpc_letters:
        return "There is not enough info"
    else:
        return cpc_letters

# Function to preprocess text
def preprocess_text(text):
    text = remove_html_tags(text)
    text = text.lower()
    return text

# Function to interpret predictions
def interpret_predictions(predictions, threshold=0.5):
    return (predictions > threshold).astype(int)

# Function to extract features and importance
def extract_features_and_importance(text):
    inputs = tokenizer_bert(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model_bert(**inputs)

    token_embeddings = outputs.last_hidden_state.squeeze().numpy()
    token_importance = np.linalg.norm(token_embeddings, axis=1)

    tokens = tokenizer_bert.convert_ids_to_tokens(inputs['input_ids'].squeeze().numpy())
    tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]

    sentences = sent_tokenize(text)
    sentence_importance = []
    for sentence in sentences:
        sentence_tokens = tokenizer_bert.tokenize(sentence)
        importance = sum(token_importance[tokens.index(token)] for token in sentence_tokens if token in tokens)
        sentence_importance.append(importance)

    return tokens, token_importance[:len(tokens)], sentences, sentence_importance

# Function to highlight text by importance
def highlight_text_by_importance(text, tokens, token_importance):
    cmap = LinearSegmentedColormap.from_list("highlight_cmap", ["white", "lightcoral", "red"])
    norm = plt.Normalize(vmin=min(token_importance), vmax=max(token_importance))

    words = text.split()
    importances = [0] * len(words)
    token_importance_dict = dict(zip(tokens, token_importance))

    for i, word in enumerate(words):
        token = word.lower()
        lemmatized_token = lemmatizer.lemmatize(token)
        if lemmatized_token in stop_words:
            importances[i] = 0
        elif token in token_importance_dict:
            importances[i] = token_importance_dict[token]

    fig, ax = plt.subplots()
    ax.axis('off')

    max_width = 4
    current_line = 0

    x_text = 0.05
    y_text = 0.7

    line_width = 0.13
    line_height = 0.3

    for i, (word, importance) in enumerate(zip(words, importances)):
        word_width = len(word) / 25 + line_width

        if x_text + word_width > max_width:
            x_text = 0.0
            y_text -= line_height
            current_line += 1

        color = cmap(norm(importance))
        ax.text(x_text, y_text, word + ' ', color='black', backgroundcolor=color, fontsize=30, ha='left', va='center')
        x_text += word_width

    plt.tight_layout()
    st.pyplot(fig)

def predict(text, model, tokenizer, max_len):
    ids, mask, token_type_ids = prepare_text(text, tokenizer, max_len)
    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)
        predictions = torch.sigmoid(outputs).cpu().numpy()
    return predictions

@st.cache_resource
def load_pegasus_model():
    model_name = "google/pegasus-large"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

pegasus_tokenizer, pegasus_model = load_pegasus_model()


def shorten_text(text, max_length=1024):
    if not text.strip():
        return ""

    # Split text into sentences
    sentences = sent_tokenize(text)

    # Extract tokens, token importances, sentences, and sentence importances
    tokens, token_importance, _, _ = extract_features_and_importance(text)

    # Sort sentences by their importance
    sorted_indices = np.argsort(token_importance)[::-1]
    ranked_sentences = [sentences[i] for i in sorted_indices if i < len(sentences)]

    # Select sentences until the text is short enough
    short_text = ""
    current_length = 0
    for sentence in ranked_sentences:
        if current_length + len(sentence) < max_length:
            short_text += " " + sentence
            current_length += len(sentence)
        else:
            break

    return short_text


def summarize_text(text):
    if not text.strip():
        return ""

    short_text = shorten_text(text)
    inputs = pegasus_tokenizer.encode(short_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = pegasus_model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary




def get_category_description(predicted_letter):
    return categories.get(predicted_letter, "Unknown Category")

categories = {
    "A": "HUMAN NECESSITIES",
    "B": "PERFORMING OPERATIONS; TRANSPORTING",
    "C": "CHEMISTRY; METALLURGY",
    "D": "TEXTILES; PAPER",
    "E": "FIXED CONSTRUCTIONS",
    "F": "MECHANICAL ENGINEERING; LIGHTING; HEATING; WEAPONS; BLASTING",
    "G": "PHYSICS",
    "H": "ELECTRICITY"
}

df_categories = pd.DataFrame(list(categories.items()), columns=["Letter", "Category"])

method = st.selectbox("Choisissez la méthode de prédiction:", ["Prédiction en insérant du texte", "Prédiction en téléchargeant un fichier CSV"])

if method == "Prédiction en insérant du texte":
    st.title("Prédiction en insérant directement une claim")
    user_input = st.text_area("Enter the text for classification:")
    if st.button("Submit"):
        if user_input:
            input_df = pd.DataFrame({'text': [user_input]})
            input_clean = remove_html_tags(input_df['text'])
            content = input_clean.iloc[0]
            predicted_cpc = preprocess_and_predict(content)
            
            predicted_cpc_letter = predicted_cpc[0]
            predicted_cpc_description = get_category_description(predicted_cpc_letter)
            st.write("### Predicted CPC:")
            st.write(f"{predicted_cpc_letter} - {predicted_cpc_description}")

            with st.expander("Generated explanation for the prediction"):
                st.write("Highlighting words that contributed to the classifier's prediction:")
                tokens, token_importance, _, _ = extract_features_and_importance(remove_html_tags(input_df['text'][0]))
                highlight_text_by_importance(remove_html_tags(input_df['text'][0]), tokens, token_importance)
            with st.expander("Résumer le texte du brevet"):
                st.write("Résumé : ")
                summary = summarize_text(input_df['text'][0])
                st.write(summary)
        else:
            st.error("Please enter some text for classification.")
        # Affichage du résumé 
            


elif method == "Prédiction en téléchargeant un fichier CSV":
    st.title('Prédiction du CPC en téléchargeant un fichier CSV')

    fichier_csv = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

    if fichier_csv is not None:
        df = pd.read_csv(fichier_csv, low_memory=True, usecols=["titre", "id", "claim"], index_col="id")
        
        st.write("Data Table (First 5 rows):", df.head())
        
        id_recherche = st.number_input('Entrez l\'ID à rechercher', min_value=int(df.index.min()), max_value=int(df.index.max()), step=1)
        
        ligne = df.loc[[id_recherche]]
        st.write("Search Result:", ligne)
        
        if not ligne.empty:
            claim_html = ligne['claim'].values[0]
            soup = BeautifulSoup(claim_html, "html.parser")
            claim_text = soup.get_text(separator=" ", strip=True)
            predicted_cpc = preprocess_and_predict(claim_text)
            
            if st.checkbox("Afficher/Masquer la claim"):
                st.markdown(f"<div style='text-align: justify;'>{claim_text}</div>", unsafe_allow_html=True)

            st.write("### Predicted CPC : ")  
            for letter in predicted_cpc: 
                predicted_cpc_description = get_category_description(letter)
                st.write(f"#### {letter} - {predicted_cpc_description}")
                
            with st.expander("Generated explanation for the prediction"):
                st.write("Highlighting words that contributed to the classifier's prediction:")
                tokens, token_importance, _, _ = extract_features_and_importance(claim_text)
                highlight_text_by_importance(claim_text, tokens, token_importance)

            # Affichage du résumé 
            with st.expander("Résumer le texte du brevet"):
                st.write("Résumé : ")
                summary = summarize_text(claim_text)
                st.write(summary)
