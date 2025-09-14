import google.generativeai as genai
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_synthetic_data(prompt, num_samples=100):
    all_texts = []
    for i in range(num_samples // 50): 
        try:
            print(f"Generating batch {i+1}...")
            response = model.generate_content(prompt)
            texts = [line.strip() for line in response.text.split('\n') if line.strip()]
            all_texts.extend(texts)
            time.sleep(10) 
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(20)
    return all_texts


# Tạo dữ liệu tiêu cực (label=1)
prompt_depressive = "Hãy đóng vai một người đang cảm thấy buồn bã, cô đơn và mệt mỏi, có vấn đề tâm lý đang cần giao tiếp. Viết 50 câu văn để giao tiếp, trò chuyện một cách tự nhiên mô tả trạng thái của bạn, hoặc kể về suy nghĩ, tâm trạng. Chỉ trả về cho tôi câu trả lời, không nói thêm gì, mỗi câu văn nằm ở một dòng, không đánh số thứ tự các dòng."
depressive_texts = generate_synthetic_data(prompt_depressive, num_samples=50) 
df_depressive = pd.DataFrame({'text': depressive_texts, 'label': 1})

# Tạo dữ liệu trung tính (label=0)
prompt_neutral = "Hãy đóng vai một người đang bình thường, không có vấn đề tâm lý gì. Viết 50 câu văn để giao tiếp, trò chuyện một cách tự nhiên kể về công việc, sở thích hoặc một hoạt động nào đó, hoặc kể về suy nghĩ, tâm trạng. Chỉ trả về cho tôi câu trả lời, không nói thêm gì, mỗi câu văn nằm ở một dòng, không đánh số thứ tự các dòng."
neutral_texts = generate_synthetic_data(prompt_neutral, num_samples=50) 
df_neutral = pd.DataFrame({'text': neutral_texts, 'label': 0})

# Kết hợp và lưu lại
final_df = pd.concat([df_depressive, df_neutral], ignore_index=True)
final_df = final_df.sample(frac=1).reset_index(drop=True) 
final_df.to_csv('synthetic_data.csv', index=False)

print(f"Đã tạo thành công file synthetic_data.csv với {len(final_df)} mẫu.")