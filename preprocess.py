# data_loader.py

import pandas as pd
import re

def clean_text(text):
    """Hàm tiền xử lý văn bản cơ bản."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r'http\S+', '', text)  # Xóa URL
    text = re.sub(r'[^\w\s]', '', text)  # Xóa dấu câu
    text = text.strip()
    return text

def load_and_preprocess_data(filepath):
    """Đọc file CSV và áp dụng tiền xử lý."""
    try:
        df = pd.read_csv(filepath)
        df.dropna(subset=['text', 'label'], inplace=True)  # Xóa các dòng thiếu dữ liệu ở cột cần thiết
        df['cleaned_text'] = df['text'].apply(clean_text)
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn '{filepath}'.")
        return pd.DataFrame(columns=['text', 'label', 'cleaned_text'])

if __name__ == '__main__':
    # Tạo một file data mẫu để chạy thử nếu chưa có
    try:
        pd.read_csv('stress_dataset.csv')
    except FileNotFoundError:
        print("Không tìm thấy 'dataset.csv', tạo file mẫu...")
        sample_data = {
            'text': [
                "Dạo này tôi chẳng muốn làm gì cả, mọi thứ thật vô nghĩa.",
                "Hôm nay tôi cảm thấy rất vui và tràn đầy năng lượng.",
                "Tôi lo lắng về bài thuyết trình ngày mai quá.",
                "Thật bực mình khi phải chờ đợi lâu như vậy.",
                "Tôi chỉ muốn ngủ cả ngày thôi, không muốn gặp ai cả.",
                "Tôi thất vọng về kết quả của mình.",
                "Cảm giác trống rỗng và cô đơn quá.",
                "Sao mọi thứ lại tồi tệ đến mức này, tôi không muốn tiếp tục nữa."
            ],
            'label': [5, 0, 1, 2, 4, 3, 4, 6]
        }
        df_sample = pd.DataFrame(sample_data)
        df_sample.to_csv('dataset.csv', index=False)

    # Chạy thử để kiểm tra
    data = load_and_preprocess_data('stress_dataset.csv')
    print("Dữ liệu sau khi xử lý:")
    print(data.head())