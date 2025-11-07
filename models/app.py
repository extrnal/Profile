from flask import Flask, render_template, request
import pickle
import pandas as pd
import re
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

app = Flask(__name__)

# Path model
MODEL_DIR = 'models'

# Load semua model SMOTE
bagging_smote = pickle.load(open(os.path.join(MODEL_DIR, 'bagging_model_smote.sav'), 'rb'))
xgb_smote = pickle.load(open(os.path.join(MODEL_DIR, 'XGB_model_smote.sav'), 'rb'))
ada_smote = pickle.load(open(os.path.join(MODEL_DIR, 'AdaBoost_model_smote.sav'), 'rb'))

# Load semua model TANPA SMOTE
bagging_nosmote = pickle.load(open(os.path.join(MODEL_DIR, 'bagging_model_nosmote.sav'), 'rb'))
xgb_nosmote = pickle.load(open(os.path.join(MODEL_DIR, 'XGB_model_nosmote.sav'), 'rb'))
ada_nosmote = pickle.load(open(os.path.join(MODEL_DIR, 'AdaBoost_model_nosmote.sav'), 'rb'))

# Load vectorizer
tfidf = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.sav'), 'rb'))

# Mapping label (ubah sesuai labelmu)
mapping = {
    'edukasi': 0,
    'kesehatan': 1,
    'kuliner' : 2,
    'otomotif': 3,
    'teknologi': 4
    
    
}
mapping_inv = {v: k for k, v in mapping.items()}

# Preprocessing
def preprocess_text(text):
    # 1. Case folding (lowercase)
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)  # hapus angka
    text = re.sub(r'\W+', ' ', text)  # hapus simbol
    text = re.sub(r'\s+', ' ', text).strip()  # hapus spasi berlebih

    # 2. Tokenizing
    words = word_tokenize(text)

    # 3. Stopword removal (standar dari NLTK Bahasa Indonesia)
    stop_words = stopwords.words('indonesian')
    words = [word for word in words if word not in stop_words]

    # 4. Hapus token satu huruf (contoh: 'a', 'b', 'x', dll)
    words = [word for word in words if len(word) > 1]

    # 5. Stemming dengan Sastrawi
    stemmed_words = [stemmer.stem(word) for word in words]

    # Gabungkan kembali menjadi teks
    return ' '.join(stemmed_words)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        judul = request.form.get('judul', '')
        cleaned = preprocess_text(judul)
        tfidf_vector = tfidf.transform([cleaned])

        # Prediksi model SMOTE
        pred_bagging_smote = bagging_smote.predict(tfidf_vector)[0]
        pred_xgb_smote = xgb_smote.predict(tfidf_vector)[0]
        pred_ada_smote = ada_smote.predict(tfidf_vector)[0]

        # Prediksi model NON SMOTE
        pred_bagging_nosmote = bagging_nosmote.predict(tfidf_vector)[0]
        pred_xgb_nosmote = xgb_nosmote.predict(tfidf_vector)[0]
        pred_ada_nosmote = ada_nosmote.predict(tfidf_vector)[0]

        result = {
            'judul': judul,
            'cleaned': cleaned,
            'smote': {
                'bagging': mapping_inv.get(pred_bagging_smote, 'Unknown'),
                'xgb': mapping_inv.get(pred_xgb_smote, 'Unknown'),
                'ada': mapping_inv.get(pred_ada_smote, 'Unknown')
            },
            'nosmote': {
                'bagging': mapping_inv.get(pred_bagging_nosmote, 'Unknown'),
                'xgb': mapping_inv.get(pred_xgb_nosmote, 'Unknown'),
                'ada': mapping_inv.get(pred_ada_nosmote, 'Unknown')
            }
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
