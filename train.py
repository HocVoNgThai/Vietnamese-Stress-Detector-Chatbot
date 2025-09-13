# train_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from preprocess import load_and_preprocess_data

# 1. Tải và xử lý dữ liệu (giả sử file của bạn tên là 'dataset.csv')
DATASET_PATH = 'stress_dataset.csv'
df = load_and_preprocess_data(DATASET_PATH)

if df.empty:
    print("Không có dữ liệu để huấn luyện. Vui lòng kiểm tra lại file dataset.csv.")
else:
    print(f"Đã tải {len(df)} mẫu từ {DATASET_PATH}")
    
    # 2. Chuẩn bị dữ liệu cho mô hình
    X = df['cleaned_text']
    y = df['label']

    # 3. Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Vector hóa văn bản
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Huấn luyện mô hình Logistic Regression (hỗ trợ đa lớp tự động)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # 6. Đánh giá mô hình
    y_pred = model.predict(X_test_vec)
    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nBáo cáo chi tiết cho từng lớp:")
    print(classification_report(y_test, y_pred))

    # 7. Lưu mô hình và vectorizer
    joblib.dump(model, 'emotion_classifier.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    print("\n✅ Mô hình đã được huấn luyện và lưu lại thành công!")
    print("   -> emotion_classifier.pkl")
    print("   -> tfidf_vectorizer.pkl")