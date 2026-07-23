import requests
import numpy as np
import joblib
import os
from src.config import OLLAMA_URL, OLLAMA_MODEL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tfidf = joblib.load(os.path.join(BASE_DIR, 'models', 'tfidf.pkl'))

def explain_classic(text):
    vec = tfidf.transform([text])
    indices = vec.indices
    values = vec.data
    if len(values) == 0:
        return ["(нет ключевых слов)"]
    # Сортируем по значениям
    sorted_idx = np.argsort(values)[::-1][:3]
    top_indices = indices[sorted_idx]
    feature_names = tfidf.get_feature_names_out()
    top_words = [feature_names[i] for i in top_indices]
    return top_words

def explain_llm(text, predicted_genre):
    prompt = f"""Книга с описанием: "{text}"
Была отнесена к жанру "{predicted_genre}".
Объясни, почему этот текст относится к данному жанру. Укажи ключевые слова или фразы из описания. Ответ напиши на русском языке, кратко (2-3 предложения)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return "Не удалось получить объяснение."
    except Exception as e:
        return f"Ошибка: {str(e)}"