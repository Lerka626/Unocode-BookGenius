import sys
from pathlib import Path
# Добавляем корень backend в путь, чтобы работали абсолютные импорты
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.model_predict import predict_classic
from src.llm_predict import predict_llm
from src.explainer import explain_classic, explain_llm
from src.config import OLLAMA_URL, OLLAMA_MODEL

app = FastAPI(title="Unocode BookGenius", description="Классификация жанров книг с объяснениями")

# Загружаем список жанров из обработанных данных (путь относительно backend)
df = pd.read_csv('result_code/data/processed/train.csv')
genre_list = sorted(df['genre'].unique())

class TextRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    genre: str
    confidence: Optional[float] = None
    method: str  # "classic" или "llm"
    explanation: Optional[str] = None

@app.get("/")
def root():
    return {"message": "Unocode BookGenius API", "available_genres": genre_list}

@app.post("/predict/classic", response_model=PredictResponse)
def predict_classic_endpoint(req: TextRequest):
    genre, conf = predict_classic(req.text)
    explanation_words = explain_classic(req.text)
    explanation = f"Ключевые слова: {', '.join(explanation_words)}"
    return PredictResponse(genre=genre, confidence=conf, method="classic", explanation=explanation)

@app.post("/predict/llm", response_model=PredictResponse)
def predict_llm_endpoint(req: TextRequest):
    genre = predict_llm(req.text, genre_list)
    explanation = explain_llm(req.text, genre)
    return PredictResponse(genre=genre, confidence=None, method="llm", explanation=explanation)

@app.post("/predict/ensemble", response_model=PredictResponse)
def predict_ensemble(req: TextRequest):
    genre_classic, conf = predict_classic(req.text)
    if conf >= 0.6:
        explanation_words = explain_classic(req.text)
        explanation = f"Классическая модель уверена на {conf:.2f}. Ключевые слова: {', '.join(explanation_words)}"
        return PredictResponse(genre=genre_classic, confidence=conf, method="classic", explanation=explanation)
    else:
        genre_llm = predict_llm(req.text, genre_list)
        explanation = explain_llm(req.text, genre_llm)
        return PredictResponse(genre=genre_llm, confidence=None, method="llm", explanation=explanation)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)