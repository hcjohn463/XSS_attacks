from transformers import AutoTokenizer, AutoModel
import faiss
import torch
import numpy as np
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 🔹 1. 設定 NLP 模型
# model_name = "BAAI/bge-small-en"  
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = 'microsoft/codebert-base'
# model_name = "jackaduma/SecBERT"
# model_name = "cssupport/mobilebert-sql-injection-detect"
# model_name = "roberta-base-openai-detector"




print(f"🔍 使用模型: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model_filename = model_name.replace('-', '_').replace('/', '_')

# 🔹 3. 加載 FAISS 向量索引 & 標籤
base_vector_dir = "../../dataset/vector"
model_vector_dir = os.path.join(base_vector_dir, model_filename)

index_file = os.path.join(model_vector_dir, f"xss_vector_index_{model_filename}.faiss")
labels_file = os.path.join(model_vector_dir, f"xss_labels_{model_filename}.npy")
payloads_file = os.path.join(model_vector_dir, f"xss_payloads_{model_filename}.npy")

# 🔹 3. 加載 FAISS 向量索引 & 標籤
print(f"📥 加載 XSS 向量庫（{index_file}）...")
index = faiss.read_index(index_file)
labels = np.load(labels_file)
payloads = np.load(payloads_file, allow_pickle=True)

print(f"✅ 向量索引中包含 {index.ntotal} 條 XSS Payloads。")

# 🔹 4. 定義 XSS Payload 嵌入函數
def get_embedding(text):
    """
    轉換 XSS Payload 成向量
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    sentence_embedding = hidden_states.mean(dim=1).squeeze().numpy()
    return sentence_embedding

# 🔹 5. 定義 XSS 檢測函數
def classify_xss_risk(user_input, k=5):
    """
    檢索 XSS 風險，並判斷是否為非法 XSS 攻擊。
    
    Args:
        user_input (str): 用戶輸入的 HTML/JavaScript 內容。
        k (int): 返回的最相似 XSS Payload 數量。
    
    Returns:
        dict: 包含 XSS 風險評估結果與詳細檢索資訊。
    """
    start_time = time.perf_counter()
    print(f"\n🛠️ 檢測 XSS 風險: {user_input}\n")
    
    # 嵌入用戶輸入
    input_embedding = get_embedding(user_input)
    
    # 檢索 FAISS
    input_embedding = input_embedding / np.linalg.norm(input_embedding, keepdims=True)
    distances, indices = index.search(np.array([input_embedding], dtype="float32"), k)

    # 計算分數
    scores = {0: 0, 1: 0}
    valid_results = []
    for idx, dist in zip(indices[0], distances[0]):
        scores[labels[idx]] += dist
        valid_results.append({
            "index": int(idx), 
            "label": int(labels[idx]),  # 0=合法, 1=非法
            "distance": round(float(dist), 4),
            "payload": payloads[idx]  # XSS Payload
        })
    
    # 判斷合法性
    threshold = 0.7  # 設定相似度門檻
    max_score = max(scores[0], scores[1])  # 取最高的相似度分數

    if max_score < threshold:
        legality = "合法"  # 如果最高相似度低於閾值，判定為合法
    else:
        legality = "非法" if scores[1] > scores[0] else "合法"

    # 計算推論時間
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        "input_payload": user_input,
        "legality": legality,
        "reason": f"Scores: {{'合法': {scores[0]:.4f}, '非法': {scores[1]:.4f}}}",
        "details": valid_results,
        "inference_time_ms": inference_time_ms
    }

# 🔹 6. 設置 `k` 值
k_value = 3

# 🔹 7. 循環輸入測試 XSS Payload
while True:
    user_query = input("請輸入 XSS Payload（或輸入 'exit' 結束）：")
    if user_query.lower() == 'exit':
        print("🚀 結束 XSS 檢測程序。")
        break

    result = classify_xss_risk(user_query, k=k_value)

    # 🔹 8. 輸出結果
    print("\n📊 檢測結果：")
    print(f"📝 輸入 Payload: {user_query}")
    print(f"🔍 判斷結果: {result['legality']}")
    print(f"🛠️ 判斷依據: {result['reason']}")
    print(f"⏱️ 推論時間: {result['inference_time_ms']:.4f} ms")
    print(f"\n🔍 最相似的 {k_value} 個 XSS Payloads：")
    for i, res in enumerate(result["details"], start=1):
        print(f"{i}. [索引 {res['index']}] XSS: {res['payload']} (標籤: {res['label']}, 距離: {res['distance']})")

print("\n✅ XSS 檢測完成，程序結束！🚀")
