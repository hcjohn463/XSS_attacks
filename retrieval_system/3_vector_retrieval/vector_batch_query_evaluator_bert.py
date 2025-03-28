import os
import csv
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 🔹 1. 選擇嵌入模型（建議用 BGE-M3 或 Sentence-BERT）
# model_name = "BAAI/bge-small-en"  
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = 'microsoft/codebert-base'
# model_name = "jackaduma/SecBERT"
# model_name = "cssupport/mobilebert-sql-injection-detect"
# model_name = "roberta-base-openai-detector"

testing = "XSS_dataset_testing_13636"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)  # 🚀 把模型移動到 GPU

model_filename = model_name.replace('-', '_').replace('/', '_')

print(f"正在使用 {model_name} 模型進行 XSS 檢測...")

# 🔹 2. 設定 XSS 向量資料庫目錄
base_output_dir = "D:/RAG/xss_attacks/result/retrieval"
os.makedirs(base_output_dir, exist_ok=True)

# 🔹 3. 加載 FAISS 向量索引 & 標籤
base_vector_dir = "D:/RAG/xss_attacks/dataset/vector"
model_vector_dir = os.path.join(base_vector_dir, model_filename)

index_file = os.path.join(model_vector_dir, f"xss_vector_index_{model_filename}.faiss")
labels_file = os.path.join(model_vector_dir, f"xss_labels_{model_filename}.npy")
payloads_file = os.path.join(model_vector_dir, f"xss_payloads_{model_filename}.npy")

print(f"📥 加載 XSS 向量庫（{index_file}）...")
index = faiss.read_index(index_file)
labels = np.load(labels_file)
payloads = np.load(payloads_file, allow_pickle=True)

print(f"✅ 向量索引中包含 {index.ntotal} 條 XSS Payloads。")


# 🔹 4. 定義 XSS Payload 嵌入函數
def get_embedding(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,  # 截斷過長的輸入
        max_length=512  # 限制最大長度為 512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    sentence_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()  
    return sentence_embedding


# 🔹 5. 定義 XSS 檢測函數
def classify_xss_risk(user_input, k=5):
    """
    判斷 XSS Payload 的風險。
    """
    # 嵌入用戶輸入
    input_embedding = get_embedding(user_input)

    # 查詢向量正規化
    normalized_query = input_embedding / np.linalg.norm(input_embedding, keepdims=True)

    # 檢索 FAISS
    distances, indices = index.search(np.array([normalized_query], dtype="float32"), k)

    # 計算分數
    scores = {0: 0, 1: 0}  # 0 = benign (合法), 1 = malicious (惡意)
    valid_results = []
    for idx, dist in zip(indices[0], distances[0]):
        scores[labels[idx]] += dist
        valid_results.append({
            "index": int(idx),
            "label": int(labels[idx]),
            "distance": round(float(dist), 4),
            "payload": payloads[idx]
        })
    
    # 判斷語句合法性
    classification = "benign" if scores[0] > scores[1] else "malicious"

    return {
        "input_payload": user_input,
        "classification": classification,
        "reason": f"Scores: {{'benign': {scores[0]:.4f}, 'malicious': {scores[1]:.4f}}}",
        "details": valid_results
    }

# 🔹 6. 讀取測試數據
input_file = f"D:/RAG/xss_attacks/dataset/{testing}.csv"
print(f"📥 讀取測試數據: {input_file}...")
with open(input_file, "r", encoding="ISO-8859-1") as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)
    print(f"✅ 共讀取到 {len(data)} 筆 XSS 測試數據。")

# 🔹 7. 設定不同 `k` 值測試
all_results = []
for k_value in range(1, 6):
    print(f"🔍 正在測試 k = {k_value} ...")

    # 設置輸出資料夾
    model_output_dir = os.path.join(base_output_dir, model_filename, f"k_{k_value}")
    os.makedirs(model_output_dir, exist_ok=True)

    # 設定輸出文件
    output_file = os.path.join(model_output_dir, f"testing_results_k_{k_value}.csv")
    confusion_matrix_file = os.path.join(model_output_dir, f"confusion_matrix_k_{k_value}.png")
    summary_file = os.path.join(base_output_dir, model_filename, "summary_results.txt")

    results = []
    true_labels = []
    predicted_labels = []

    start_time = time.time()

    # 處理每筆 XSS 測試數據
    for row in tqdm(data, desc="處理測試數據", unit="筆"):
        user_payload = row["Payload"]
        true_label = int(row["Label"])  # 0 = 合法, 1 = 惡意

        # 判斷 XSS 風險
        result = classify_xss_risk(user_payload, k=k_value)

        # 轉換標籤格式
        mapped_label = {"benign": 0, "malicious": 1}

        results.append({
            "payload": user_payload,
            "true_label": true_label,
            "predicted_label": mapped_label[result["classification"]],
            "reason": result["reason"]
        })
        true_labels.append(true_label)
        predicted_labels.append(mapped_label[result["classification"]])

    # 計算時間
    total_time = time.time() - start_time
    average_time = (total_time / len(data)) * 1000  # ms

    # 計算 Accuracy, Precision, Recall
    accuracy = accuracy_score(true_labels, predicted_labels) * 100
    precision = precision_score(true_labels, predicted_labels) * 100
    recall = recall_score(true_labels, predicted_labels) * 100

    all_results.append({
        "k": k_value,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "total_time": total_time,
        "average_time": average_time
    })

    # 存入 CSV
    print(f"📄 寫入結果到 {output_file}...")
    with open(output_file, "w", newline="", encoding="utf-8", errors="replace") as csvfile:
        fieldnames = ["payload", "true_label", "predicted_label", "reason"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # **繪製混淆矩陣**
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malicious"])
    disp.plot(cmap=plt.cm.Blues, colorbar=False, values_format='.0f')
    plt.title(f"XSS Detection: {model_name} - k = {k_value}")
    plt.savefig(confusion_matrix_file)
    print(f"✅ 混淆矩陣已保存: {confusion_matrix_file}")

# 🚀 儲存 summary 結果
with open(summary_file, "w", encoding="utf-8") as f:
    for result in all_results:
        f.write(f"k = {result['k']}\n")
        f.write(f"Accuracy: {result['accuracy']:.3f}%\n")
        f.write(f"Precision: {result['precision']:.3f}%\n")
        f.write(f"Recall: {result['recall']:.3f}%\n")
        f.write(f"Total Time: {result['total_time']:.2f}s\n")
        f.write(f"Average Time: {result['average_time']:.2f}ms\n\n")

print(f"✅ 所有結果已保存到 {summary_file}！🚀")
