import csv
import random

negative_filepath = 'negative.txt' 
positive_filepath = 'positive.txt' 
combined_filepath = 'stress_dataset.csv'
all_samples = []

print("Bắt đầu quá trình kết hợp dữ liệu...")

try:
    with open(negative_filepath, 'r', encoding='utf-8') as f:
        negative_samples = [line.strip() for line in f if line.strip()]
        all_samples.extend(negative_samples)
        print(f"Đã đọc thành công {len(negative_samples)} mẫu từ file '{negative_filepath}'.")
except FileNotFoundError:
    print(f"[LỖI] Không tìm thấy file '{negative_filepath}'. Vui lòng kiểm tra lại tên file.")

try:
    with open(positive_filepath, 'r', encoding='utf-8') as f:
        positive_samples = [line.strip() for line in f if line.strip()]
        all_samples.extend(positive_samples)
        print(f"Đã đọc thành công {len(positive_samples)} mẫu từ file '{positive_filepath}'.")
except FileNotFoundError:
    print(f"[LỖI] Không tìm thấy file '{positive_filepath}'. Vui lòng kiểm tra lại tên file.")

if all_samples:
    print(f"\nTổng cộng có {len(all_samples)} mẫu dữ liệu.")
    print("Đang xáo trộn dữ liệu...")
    random.shuffle(all_samples)
    print("Xáo trộn thành công!")

    try:
        with open(combined_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label']) 

            for sample in all_samples:
                parts = sample.rsplit(',', 1)
                if len(parts) == 2:
                    text = parts[0].strip()
                    label = parts[1].strip()
                    writer.writerow([text, label])
        
        print(f"\nĐã lưu thành công toàn bộ dữ liệu vào file '{combined_filepath}'.")
        print("Quá trình hoàn tất!")
    except IOError:
        print(f"[LỖI] Không thể ghi dữ liệu vào file '{combined_filepath}'.")
else:
    print("\nKhông có dữ liệu để xử lý. Vui lòng kiểm tra lại các file đầu vào.")