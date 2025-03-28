import os
import pandas as pd
import json
import numpy as np
import torch
import faiss
import multiprocessing
import time
import re
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# 確定設備
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# model_name = "microsoft/codebert-base"
# model_name = "jackaduma/SecBERT"
# model_name = "cssupport/mobilebert-sql-injection-detect"
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model_name = "roberta-base-openai-detector"
model_name = "BAAI/bge-small-en"

model_filename = model_name.replace('-', '_').replace('/', '_')
total_samples = 200  # 每類各取 total_samples 筆，共 400 筆作為訓練資料來源

# 設定參數
input_file = "D:/RAG/XSS_attacks/dataset/XSS_dataset.csv"
dataset_dir = f"D:/RAG/XSS_attacks/dataset/{total_samples}"
os.makedirs(dataset_dir, exist_ok=True)
json_dir = "D:/RAG/XSS_attacks/dataset/json"
vector_dir = "D:/RAG/XSS_attacks/dataset/vector"
retrieval_dir = f"D:/RAG/XSS_attacks/result/retrieval/{total_samples}"
os.makedirs(json_dir, exist_ok=True)
os.makedirs(vector_dir, exist_ok=True)
os.makedirs(retrieval_dir, exist_ok=True)

def clean_excel_text(s):
    if isinstance(s, str):
        return re.sub(r'[\x00-\x1F]+', '', s)
    return s

def split_dataset():
    df = pd.read_csv(input_file, encoding="utf-8")
    benign_pool = df[df["Label"] == 0].sample(n=total_samples, random_state=42)
    attack_pool = df[df["Label"] == 1].sample(n=total_samples, random_state=42)
    combined = pd.concat([benign_pool, attack_pool])
    test_df = df.drop(combined.index)  # 減去訓練來源

    # 清除非法 Excel 字元並儲存成 excel
    combined_cleaned = combined.applymap(clean_excel_text)
    test_df_cleaned = test_df.applymap(clean_excel_text)
    combined_cleaned.to_excel(os.path.join(dataset_dir, "xss_dataset_combined.xlsx"), index=False)
    test_df_cleaned.to_excel(os.path.join(dataset_dir, "xss_dataset_testing.xlsx"), index=False)

    print(f"✅ 測試集（{len(test_df)}）已儲存！")
    return benign_pool.reset_index(drop=True), attack_pool.reset_index(drop=True), test_df

def convert_queries_to_vectors(train_data, index_id):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    payloads = train_data["Payload"].tolist()
    labels = train_data["Label"].tolist()

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    embeddings = np.array([get_embedding(text) for text in tqdm(payloads, desc="轉換向量")])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    index_file = os.path.join(vector_dir, f"xss_vector_index_{index_id}.faiss")
    label_file = os.path.join(vector_dir, f"xss_labels_{index_id}.npy")

    faiss.write_index(faiss_index, index_file)
    np.save(label_file, labels)
    print(f"✅ 向量庫 {index_id} 已建立並儲存！")

def precompute_test_embeddings(test_df):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    embeddings = np.array([get_embedding(payload) for payload in tqdm(test_df["Payload"], desc="轉換測試向量")])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings, test_df["Label"].tolist()

def evaluate_vectors(test_df, test_embeddings, test_labels, benign_pool, attack_pool):

    def classify_xss_risk(input_embedding, k, index_id):
        index = faiss.read_index(os.path.join(vector_dir, f"xss_vector_index_{index_id}.faiss"))
        labels = np.load(os.path.join(vector_dir, f"xss_labels_{index_id}.npy"))
        distances, indices = index.search(np.array([input_embedding], dtype="float32"), k)
        scores = {0: 0, 1: 0}
        for idx, dist in zip(indices[0], distances[0]):
            scores[labels[idx]] += dist
        return "benign" if scores[0] > scores[1] else "malicious"

    results = []
    for malicious_count in range(total_samples + 1):
        legit_count = total_samples - malicious_count
        train_part = pd.concat([
            attack_pool.iloc[:malicious_count],
            benign_pool.iloc[:legit_count]
        ]).reset_index(drop=True)

        convert_queries_to_vectors(train_part, index_id=malicious_count)

        predicted_labels = [
            1 if classify_xss_risk(emb, k=1, index_id=malicious_count) == "malicious" else 0
            for emb in tqdm(test_embeddings, desc=f"訓練比例 {malicious_count}/{total_samples}")
        ]

        accuracy = accuracy_score(test_labels, predicted_labels) * 100
        precision = precision_score(test_labels, predicted_labels, zero_division=0) * 100
        recall = recall_score(test_labels, predicted_labels, zero_division=0) * 100
        total_time = round(time.time(), 3)
        avg_time = round((total_time / len(test_labels)) * 1000, 3)

        results.append([malicious_count, legit_count, accuracy, precision, recall, total_time, avg_time])

    df = pd.DataFrame(results, columns=["Malicious", "Legit", "Accuracy", "Precision", "Recall", "Total Time (s)", "Average Time (ms)"])
    df.to_csv(os.path.join(retrieval_dir, f"XSS_summary_results_{model_filename}.csv"), index=False)
    print("✅ 測試結果已完成並儲存！")

def main():
    benign_pool, attack_pool, test_df = split_dataset()
    test_embeddings, test_labels = precompute_test_embeddings(test_df)
    evaluate_vectors(test_df, test_embeddings, test_labels, benign_pool, attack_pool)
    print("🚀 全部流程完成！")

if __name__ == "__main__":
    main()
