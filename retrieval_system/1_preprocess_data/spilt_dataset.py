import pandas as pd
from sklearn.model_selection import train_test_split

# 🔹 1. 讀取 XSS 資料集
input_file = "D:/RAG/XSS_attacks/dataset/XSS_dataset.csv"
df = pd.read_csv(input_file)

# 🔹 2. 設定訓練集比例（可以調整）
train_ratio = 0.8  # 80% 訓練，20% 測試

# 🔹 3. 切割數據集（確保隨機分配）
train_df, test_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42, stratify=df["Label"])

# 🔹 4. 設定輸出檔案
train_file = "D:/RAG/XSS_attacks/dataset/XSS_dataset_training.csv"
test_file = "D:/RAG/XSS_attacks/dataset/XSS_dataset_testing.csv"

# 🔹 5. 儲存分割後的數據集
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"✅ 訓練集已保存至: {train_file} (共 {len(train_df)} 筆)")
print(f"✅ 測試集已保存至: {test_file} (共 {len(test_df)} 筆)")
