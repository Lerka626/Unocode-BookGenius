import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'result_code', 'data', 'processed', 'train.csv')
    test_path = os.path.join(base_dir, 'result_code', 'data', 'processed', 'test.csv')
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    X_train = train['description'].astype(str)
    y_train = train['genre']
    X_test = test['description'].astype(str)
    y_test = test['genre']
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train_tfidf, y_train_enc)
    
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test_enc, preds)
    print(f"Accuracy на тесте: {acc:.4f}")
    print(classification_report(y_test_enc, preds, target_names=le.classes_))
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'lgb_model.pkl'))
    joblib.dump(tfidf, os.path.join(models_dir, 'tfidf.pkl'))
    joblib.dump(le, os.path.join(models_dir, 'label_encoder.pkl'))
    
    with open(os.path.join(models_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
    
    print("Модель сохранена.")

if __name__ == "__main__":
    train()