import os
import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch

# 🔹 1. 選擇 NLP 模型（建議用 BGE-M3 或 Sentence-BERT）
model_name = "BAAI/bge-small-en"  # 或 "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 🔹 2. 設定 XSS 向量資料庫目錄
base_dir = "D:/RAG/xss_attacks/dataset/vector"
model_dir = os.path.join(base_dir, model_name.replace('-', '_').replace('/', '_'))

# 設定 FAISS 路徑
index_file = os.path.join(model_dir, "xss_vector_index.faiss")
labels_file = os.path.join(model_dir, "xss_labels.npy")
payloads_file = os.path.join(model_dir, "xss_payloads.npy")

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

# 🔹 5. 定義 XSS 檢索函數
def retrieve_xss_risk(user_input, k=3):
    """
    檢索 XSS 風險，返回最相似的攻擊字串。
    
    Args:
        user_input (str): 用戶輸入的 HTML/JavaScript 內容。
        k (int): 返回的最相似 XSS Payload 數量。
    
    Returns:
        list: 包含檢索到的索引、標籤、距離和 XSS Payload。
    """
    print(f"\n🛠️ 檢測 XSS 風險: {user_input}")
    
    # 嵌入用戶輸入
    input_embedding = get_embedding(user_input)
    
    # 檢索 FAISS
    distances, indices = index.search(np.array([input_embedding], dtype="float32"), k)
    
    # 返回結果
    results = []
    print(f"\n🔍 最相似的 {k} 個 XSS Payloads：")
    for i, idx in enumerate(indices[0]):
        result = {
            "index": int(idx),
            "label": int(labels[idx]),  # 0 = 合法, 1 = 非法
            "distance": float(distances[0][i]),
            "payload": payloads[idx]  # XSS Payload
        }
        results.append(result)
        print(f"- XSS Payload: {result['payload']}, 標籤: {result['label']}, 距離: {result['distance']}")
    
    return results

# 🔹 6. 測試 XSS 檢索
test_input = "<img src=x onerror=alert('XSS')>"
result = retrieve_xss_risk(test_input, k=3)

# 🔹 7. 打印詳細結果
print("\n📊 詳細結果：")
for i, res in enumerate(result, start=1):
    print(f"第 {i} 筆：")
    print(f"  - 索引: {res['index']}")
    print(f"  - 標籤: {res['label']} (0=合法, 1=非法)")
    print(f"  - 距離: {res['distance']}")
    print(f"  - 相似 XSS Payload: {res['payload']}")

print("\n✅ XSS 檢測完成！🚀")
