from transformers import AutoTokenizer, AutoModel
import faiss
import torch
import numpy as np
import time
import os

# 嵌入模型名稱
model_name = "microsoft/codebert-base"
# model_name = "cssupport/mobilebert-sql-injection-detect"
# model_name = "jackaduma/SecBERT"

print(f"模型名稱: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 文件名轉換（替換 - 為 _）
model_file_name = model_name.replace('-', '_').replace('/', '_')

# 動態設置文件路徑
base_dir = "D:/RAG/SQL_legality/dataset/vector"
model_dir = os.path.join(base_dir, model_file_name)

index_file = os.path.join(model_dir, f"vector_index_{model_file_name}.faiss")
labels_file = os.path.join(model_dir, f"vector_labels_{model_file_name}.npy")
queries_file = os.path.join(model_dir, f"queries_{model_file_name}.npy")

# 加載向量索引和標籤
print(f"加載模型 {model_name} 的向量資料...")
index = faiss.read_index(index_file)
labels = np.load(labels_file)
queries = np.load(queries_file, allow_pickle=True)

print(f"向量索引中包含 {index.ntotal} 條語句。")

# 定義嵌入函數
def get_embedding(sentence):
    # Token 化
    inputs = tokenizer(
        sentence,
        return_tensors="pt",  # 返回 PyTorch 張量
        padding="max_length", # 補零至最大長度
        truncation=True,      # 截斷長輸入
        max_length=512        # 最大輸入長度
    )
    print(f"查詢語句: {sentence}")

    # 格式化 Input IDs 為對齊格式，並以頓號分隔
    input_ids = inputs['input_ids'][0, :20].tolist()
    formatted_input_ids = ", ".join([f"{id_:>5}" for id_ in input_ids])  # 每個數字占 5 個空間，頓號分隔
    print(f"- Input IDs: {formatted_input_ids}")

    # 嵌入生成
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取最後隱藏層，並進行平均池化
    last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_length, embedding_dim]
    sentence_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()  # 平均池化為句子嵌入
    formatted_embedding = ", ".join([f"{x:.4f}" for x in sentence_embedding[:5]])  # 格式化前 5 個嵌入值
    print(f"- 句子嵌入向量 (維度: {len(sentence_embedding)}): {formatted_embedding} ...")

    return sentence_embedding

def classify_sql_legality(user_query, k=5, epsilon=1e-6):
    start_time = time.perf_counter()
    print(f"\n輸入語句: {user_query}\n")
    
    # 嵌入用戶輸入語句
    query_embedding = get_embedding(user_query)
    
    # 查詢向量正規化
    normalized_query = query_embedding / np.linalg.norm(query_embedding, keepdims=True)
    
    # 檢索向量索引
    distances, indices = index.search(np.array([normalized_query], dtype="float32"), k)

    # 計算分數
    scores = {0: 0, 1: 0}
    valid_results = []
    for idx, dist in zip(indices[0], distances[0]):
        scores[labels[idx]] += dist
        valid_results.append({
            "index": int(idx),
            "label": int(labels[idx]),
            "distance": round(float(dist), 4),
            "query": queries[idx]
        })
    
    # 判斷語句合法性
    legality = "legal" if scores[0] > scores[1] else "illegal"
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        "input_query": user_query,
        "legality": legality,
        "reason": f"Scores: {{'legal': {scores[0]:.4f}, 'illegal': {scores[1]:.4f}}}",
        "details": valid_results,
        "inference_time_ms": inference_time_ms
    }

# 設置 k 值
k_value = 1

# 循環輸入查詢語句
while True:
    user_query = input("請輸入SQL語句 (或輸入 'exit' 結束): ")
    if user_query.lower() == 'exit':
        print("結束程序。")
        break

    result = classify_sql_legality(user_query, k=k_value)

    # 輸出結果
    print("\n判斷結果：")
    print(f"輸入語句: {user_query}")
    print(f"語句合法性：{result['legality']}")
    print(f"原因：{result['reason']}")
    print(f"推論時間: {result['inference_time_ms']:.4f} ms")
    print(f"3.2 SQL 語句合法性判斷完成，使用模型: {model_name}！")
