import os
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 選擇嵌入模型
# model_name = "BAAI/bge-small-en"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"正在使用 {model_name} 模型進行 XSS 檢測...")

# 取得嵌入向量的函數
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 取得句子嵌入（平均池化）
    hidden_states = outputs.last_hidden_state
    sentence_embedding = hidden_states.mean(dim=1).squeeze().numpy()
    return sentence_embedding

# 讀取測試數據
input_file = "D:/RAG/xss_attacks/dataset/XSS_dataset_testing_cleaned.csv"
print(f"📥 讀取測試數據: {input_file}...")
results = []
true_labels = []
predicted_labels = []

data_count = 0
with open(input_file, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)
    data_count = len(data)
    print(f"✅ 共讀取到 {data_count} 筆 XSS 測試數據。")

# 記錄整體測試時間
start_time_total = time.time()

# 計算每個 Payload 的嵌入向量，並進行分類
for row in tqdm(data, desc="處理測試數據進度", unit="筆"):
    user_payload = row["Payload"]
    true_label = int(row["Label"])  # 0 = benign, 1 = malicious

    # 取得嵌入向量
    start_time = time.perf_counter()
    payload_embedding = get_embedding(user_payload)
    inference_time_ms = (time.perf_counter() - start_time) * 1000  # 轉換為毫秒

    # 使用閾值來判斷是否為惡意 XSS
    threshold = 0.5  # 可調整
    similarity_score = np.linalg.norm(payload_embedding)  # 這裡可以換成 Cosine Similarity 計算
    predicted_label = 1 if similarity_score > threshold else 0  # 設定閾值分類

    results.append({
        "payload": user_payload,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "similarity_score": round(similarity_score, 4),
        "inference_time_ms": round(inference_time_ms, 4)
    })
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

# 記錄測試完成時間
total_time = time.time() - start_time_total
average_time = (total_time / data_count) * 1000  # 轉換為毫秒

# 設置輸出目錄
base_output_dir = "D:/RAG/xss_attacks/result/direct"
model_output_dir = os.path.join(base_output_dir, model_name.replace('-', '_').replace('/', '_'))

# 確保資料夾存在
os.makedirs(model_output_dir, exist_ok=True)

# 設置輸出文件路徑
output_file = os.path.join(model_output_dir, f"testing_results_{model_name.replace('-', '_').replace('/', '_')}.csv")
confusion_matrix_file = os.path.join(model_output_dir, f"confusion_matrix_{model_name.replace('-', '_').replace('/', '_')}.png")
summary_file = os.path.join(model_output_dir, "summary_results.txt")

# 寫入結果到 CSV
print(f"📄 寫入結果到 {output_file}...")
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["payload", "true_label", "predicted_label", "similarity_score", "inference_time_ms"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print(f"✅ 結果已保存到 {output_file}！")

# 計算 Accuracy, Precision, Recall
accuracy = accuracy_score(true_labels, predicted_labels) * 100
precision = precision_score(true_labels, predicted_labels) * 100
recall = recall_score(true_labels, predicted_labels) * 100

# 計算總時間格式化
total_minutes = int(total_time // 60)
total_seconds = int(total_time % 60)

# 打印結果
print(f"📊 Accuracy: {accuracy:.3f}%")
print(f"📊 Precision: {precision:.3f}%")
print(f"📊 Recall: {recall:.3f}%")
print(f"⏱️ Total Time: {total_minutes}min {total_seconds}sec")
print(f"⏱️ Average Time: {average_time:.2f}ms")

# 繪製混淆矩陣
print("📊 繪製混淆矩陣...")
cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malicious"])
disp.plot(cmap=plt.cm.Blues, colorbar=False, values_format='.0f')

# 設置標題與標籤
plt.title(f"Confusion Matrix_{model_name.replace('-', '_').replace('/', '_')}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# 保存混淆矩陣圖像
plt.savefig(confusion_matrix_file)
plt.show()

print(f"✅ 混淆矩陣已保存為：{confusion_matrix_file}")

# 生成 Summary File
print(f"📄 生成總結文件 {summary_file}...")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.3f}%\n")
    f.write(f"Precision: {precision:.3f}%\n")
    f.write(f"Recall: {recall:.3f}%\n")
    f.write(f"Total Time: {total_minutes}min {total_seconds}sec\n")
    f.write(f"Average Time: {average_time:.2f}ms\n")

print(f"✅ Summary 已保存至 {summary_file}！")
