import pandas as pd
import re
from underthesea import word_tokenize

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    if df.empty:
        return df
    
    def clean_text(text):
        # Loại bỏ ký tự đặc biệt, số, và xuống dòng
        text = re.sub(r'[^\w\s]', '', text.lower().strip())
        # Phân đoạn từ tiếng Việt
        text = word_tokenize(text, format="text")  # Giữ nguyên câu dạng text
        return text

    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[['cleaned_text', 'label']].dropna()
    print(df['label'].value_counts())
    return df