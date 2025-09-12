import google.generativeai as genai
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Cấu hình API key của bạn
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_synthetic_data(prompt, num_samples=100):
    """Hàm tạo dữ liệu từ LLM"""
    all_texts = []
    for i in range(num_samples // 50): 
        try:
            print(f"Generating batch {i+1}...")
            response = model.generate_content(prompt)
            # Tách các câu ra, mỗi câu trên 1 dòng
            texts = [line.strip() for line in response.text.split('\n') if line.strip()]
            all_texts.extend(texts)
            time.sleep(2) 
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)
    return all_texts


# Tạo dữ liệu tiêu cực (label=1)
prompt_depressive = "Hãy đóng vai một người đang cảm thấy buồn bã, cô đơn và mệt mỏi. Viết 50 câu văn ngắn gọn để giao tiếp, tự nhiên mô tả trạng thái của bạn."
depressive_texts = generate_synthetic_data(prompt_depressive, num_samples=1000) 
df_depressive = pd.DataFrame({'text': depressive_texts, 'label': 1})

# Tạo dữ liệu trung tính (label=0)
prompt_neutral = "Hãy đóng vai một người đang có một ngày bình thường. Viết 50 câu văn ngắn gọn để giao tiếp, tự nhiên kể về công việc, sở thích hoặc một hoạt động nào đó."
neutral_texts = generate_synthetic_data(prompt_neutral, num_samples=1000) 
df_neutral = pd.DataFrame({'text': neutral_texts, 'label': 0})

# Kết hợp và lưu lại
final_df = pd.concat([df_depressive, df_neutral], ignore_index=True)
final_df = final_df.sample(frac=1).reset_index(drop=True) 
final_df.to_csv('synthetic_data.csv', index=False)

print(f"Đã tạo thành công file synthetic_data.csv với {len(final_df)} mẫu.")