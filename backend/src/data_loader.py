import pandas as pd
import os
import kagglehub
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    # Скачиваем ваш датасет (или читаем из raw, если уже скачан)
    # Попробуем скачать через kagglehub
    try:
        path = kagglehub.dataset_download("mihikaajayjadhav/books-dataset-15k-books-across-100-categories")
        # Найдём в папке файл .csv
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError("CSV файл не найден в скачанной папке.")
        csv_path = os.path.join(path, files[0])
        df = pd.read_csv(csv_path, on_bad_lines='skip')  # на случай кривых строк
    except Exception as e:
        print(f"Не удалось скачать через kagglehub: {e}")
        print("Попробуйте скачать вручную и положить файл в backend/data/raw/google_books_dataset.csv")
        # Ручной путь
        manual_path = 'backend/data/raw/google_books_dataset.csv'
        if os.path.exists(manual_path):
            df = pd.read_csv(manual_path, on_bad_lines='skip')
        else:
            raise FileNotFoundError("Файл не найден ни в скачанной папке, ни в backend/data/raw/")
    
    print("Датасет загружен, колонки:", df.columns.tolist())
    
    # Оставляем нужные колонки (если они есть, иначе адаптируем)
    # В этом датасете колонки: description, categories, search_category
    # Если нет description, возможно, есть 'description' с маленькой буквы
    # Проверим наличие
    if 'description' not in df.columns:
        # Попробуем другие варианты
        possible_desc = [col for col in df.columns if 'desc' in col.lower()]
        if possible_desc:
            desc_col = possible_desc[0]
        else:
            raise KeyError("Нет колонки с описанием")
    else:
        desc_col = 'description'
    
    if 'categories' not in df.columns:
        # Попробуем 'category' или 'search_category'
        possible_cat = [col for col in df.columns if 'category' in col.lower()]
        if possible_cat:
            cat_col = possible_cat[0]
        else:
            raise KeyError("Нет колонки с категориями")
    else:
        cat_col = 'categories'
    
    # Берём нужные колонки
    df = df[[desc_col, cat_col]].copy()
    df.columns = ['description', 'categories_raw']  # переименовываем для удобства
    
    # Удаляем строки без описания
    df = df[df['description'].notna()]
    
    # Извлекаем первый жанр из строки (например "Fiction / Mystery" -> "Fiction")
    def extract_genre(cat_str):
        if pd.isna(cat_str):
            return None
        # Разделители: /, ,, &, |
        for sep in [' / ', ' & ', ', ', '|']:
            if sep in cat_str:
                return cat_str.split(sep)[0].strip()
        return cat_str.strip()
    
    df['genre'] = df['categories_raw'].apply(extract_genre)
    df = df[df['genre'].notna()]
    df = df[['description', 'genre']]
    
    # Оставляем топ-10 жанров
    top_genres = df['genre'].value_counts().head(10).index.tolist()
    df = df[df['genre'].isin(top_genres)]
    
    # Разбиваем на train/test
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['genre'])
    
    # Сохраняем
    os.makedirs('backend/data/processed', exist_ok=True)
    train.to_csv('backend/data/processed/train.csv', index=False)
    test.to_csv('backend/data/processed/test.csv', index=False)
    
    print(f"Обработано {len(df)} записей. Train: {len(train)}, Test: {len(test)}")
    print(f"Топ-10 жанров: {top_genres}")
    return train, test

if __name__ == "__main__":
    load_and_prepare_data()