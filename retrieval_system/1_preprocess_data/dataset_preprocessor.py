import pandas as pd
import json
import os

# 🔹 1. 讀取 CSV 文件
file_path = "D:/RAG/XSS_attacks/dataset/XSS_dataset_training.csv"  # 替換為你的 CSV 文件路徑
df = pd.read_csv(file_path)

# 🔹 2. 查看數據結構（可選）
print(df.head())

# 🔹 3. 轉換為 JSON 格式
data = df.to_dict(orient="records")

# 🔹 4. 確保 JSON 目錄存在
json_dir = "D:/RAG/XSS_attacks/dataset/json"
os.makedirs(json_dir, exist_ok=True)  # 如果資料夾不存在則創建

# 🔹 5. 設定 JSON 文件輸出路徑
json_file_path = os.path.join(json_dir, "xss_dataset_training.json")

# 🔹 6. 存成 JSON 檔案
with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"✅ JSON 文件已成功保存至: {json_file_path}")
