import os
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from tqdm import tqdm               

# 🔹 1. 選擇嵌入模型（建議用 BGE-M3 或 Sentence-BERT）
# model_name = "BAAI/bge-small-en"  
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = 'microsoft/codebert-base'
# model_name = "jackaduma/SecBERT"
# model_name = "cssupport/mobilebert-sql-injection-detect"
# model_name = "roberta-base-openai-detector"


training = "xss_dataset.json"


print(f"🔍 使用模型: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model_filename = model_name.replace('-', '_').replace('/', '_')

# 🔹 2. 設定資料與輸出目錄
base_output_dir = "../../dataset/vector"
model_output_dir = os.path.join(base_output_dir, model_name.replace('-', '_').replace('/', '_'))
os.makedirs(model_output_dir, exist_ok=True)

# 🔹 3. 讀取 XSS 資料集
with open(f"../../dataset/json/{training}", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 取得 Payloads 與 Labels
payloads = [item["Payload"] for item in dataset]  # XSS 內容
labels = [item["Label"] for item in dataset]  # 0=合法, 1=非法

# 確保數據一致
assert len(payloads) == len(labels), "❌ 數據數量不一致，請檢查資料集！"

# 🔹 4. 定義嵌入函數
def get_embedding(text):
    """
    使用 NLP 模型提取 XSS Payload 向量
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    sentence_embedding = hidden_states.mean(dim=1).squeeze().numpy()
    return sentence_embedding

# 🔹 5. 轉換所有 XSS Payload 為嵌入向量
print(f"🚀 正在使用 {model_name} 轉換 XSS Payload ...")
embeddings = np.array([get_embedding(text) for text in tqdm(payloads, desc="嵌入進度")])

# 🔹 6. 初始化 FAISS 向量庫
embedding_dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(embedding_dimension)  # 內積相似度
# 正規化嵌入向量
normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
faiss_index.add(np.array(normalized_embeddings, dtype='float32'))


# 🔹 7. 儲存 FAISS 索引
faiss_index_file = os.path.join(model_output_dir, f"xss_vector_index_{model_filename}.faiss")
faiss.write_index(faiss_index, faiss_index_file)
print(f"✅ 向量庫已儲存為: {faiss_index_file}")

# 🔹 8. 儲存標籤與原始 Payload
np.save(os.path.join(model_output_dir, f"xss_labels_{model_filename}.npy"), labels)
np.save(os.path.join(model_output_dir, f"xss_payloads_{model_filename}.npy"), payloads)

print(f"✅ 標籤與 Payload 已儲存，XSS FAISS 資料庫建立完成！🎉")
