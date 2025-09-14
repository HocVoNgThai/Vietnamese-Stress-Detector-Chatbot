import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, Features, ClassLabel, Value
import torch
import numpy as np
from underthesea import word_tokenize
import re

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    if df.empty:
        print("Không có dữ liệu để huấn luyện.")
        return df
    
    def clean_text(text):
        # Loại bỏ ký tự đặc biệt, số, xuống dòng
        text = re.sub(r'[^\w\s]', '', text.lower().strip())
        # Phân đoạn từ tiếng Việt
        text = word_tokenize(text, format="text")
        return text

    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[['cleaned_text', 'label']].dropna()
    return df

DATASET_PATH = '../Datasets/stress_dataset.csv'
df = load_and_preprocess_data(DATASET_PATH)

if df.empty:
    print("Không có dữ liệu để huấn luyện.")
else:
    print(f"Đã tải {len(df)} mẫu từ {DATASET_PATH}")
    print("Phân bố nhãn:", df['label'].value_counts())

    features = Features({
        'cleaned_text': Value('string'),
        'label': ClassLabel(num_classes=7, names=['Bình thường', 'Lo lắng', 'Bực bội', 'Thất vọng', 'Buồn', 'Mệt mỏi', 'Tuyệt vọng'])
    })
    dataset = Dataset.from_pandas(df, features=features)

    train_test = dataset.train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
    train_dataset = train_test['train']
    test_dataset = train_test['test']

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    def tokenize_function(examples):
        return tokenizer(examples['cleaned_text'], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=7)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, target_names=['Bình thường', 'Lo lắng', 'Bực bội', 'Thất vọng', 'Buồn', 'Mệt mỏi', 'Tuyệt vọng'], output_dict=True)
        return {"accuracy": acc, "classification_report": report}

    #Thay đổi thông số train nếu bạn muốn
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    trainer.train()

    results = trainer.evaluate()
    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---\n")
    print(f"Accuracy: {results['eval_accuracy']:.2f}")
    print("\nBáo cáo chi tiết:")
    for label, metrics in results['eval_classification_report'].items():
        if isinstance(metrics, dict):
            print(f"{label}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")

    model.save_pretrained("../PhoBERT_emotion_classifier")
    tokenizer.save_pretrained("PhoBERT_emotion_classifier")
    print("\nMô hình đã được huấn luyện và lưu lại thành công!")