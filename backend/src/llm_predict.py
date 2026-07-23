import requests
from src.config import OLLAMA_URL, OLLAMA_MODEL

def predict_llm(text, genre_list):
    prompt = f"""Ты — классификатор книг. Определи жанр книги по её описанию.
Возможные жанры: {', '.join(genre_list)}.
Ответь только названием жанра из этого списка, ничего другого не пиши.

Описание книги: {text}

Жанр:"""
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            # Попытаемся найти жанр в ответе
            for genre in genre_list:
                if genre.lower() in result.lower():
                    return genre
            return result  # если не нашли, возвращаем как есть
        else:
            return "Ошибка LLM"
    except Exception as e:
        return f"Ошибка: {str(e)}"