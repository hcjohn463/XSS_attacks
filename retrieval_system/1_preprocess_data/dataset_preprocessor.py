import pandas as pd
import json

# 讀取 CSV 文件
file_path = "D:/RAG/XSS_attacks/dataset/XSS_dataset.csv"  # 替換為您的 CSV 文件路徑
data = pd.read_csv(file_path)

# 查看數據結構（可選）
print(data.head())

# 讀取 Excel
df = pd.read_excel("xss_dataset.xlsx")

# 轉換為 JSON 格式
data = df.to_dict(orient="records")

# 存成 JSON 檔案
with open("xss_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)