import joblib
import numpy as np
import os

# Пути относительно backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'lgb_model.pkl')
tfidf_path = os.path.join(BASE_DIR, 'models', 'tfidf.pkl')
le_path = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)
le = joblib.load(le_path)

def predict_classic(text):
    vec = tfidf.transform([text])
    pred_enc = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    genre = le.inverse_transform([pred_enc])[0]
    confidence = np.max(proba)
    return genre, confidence