import pandas as pd
import json
import os

# **🔹 1. 讀取 XSS 數據集，確保所有非 UTF-8 字元被正確處理**
input_file = "D:/RAG/XSS_attacks/dataset/XSS_dataset.csv"
df = pd.read_csv(input_file, encoding="utf-8")  # 確保不會有編碼錯誤

# **🔹 2. 篩選出 Label = 1 (惡意) 和 Label = 0 (合法) 的樣本**
xss_attacks = df[df["Label"] == 1].sample(n = 25, random_state=42)  
benign_samples = df[df["Label"] == 0].sample(n = 25, random_state=42)  

# **🔹 3. 合併成新的訓練集**
train_df = pd.concat([xss_attacks, benign_samples])

# **🔹 4. 剩下的數據作為測試集**
test_df = df.drop(train_df.index)

# **🔹 5. 計算資料筆數**
train_size = len(train_df)
test_size = len(test_df)

# **🔹 6. 設定 CSV 輸出檔案**
train_file = f"D:/RAG/XSS_attacks/dataset/XSS_dataset_training_{train_size}.csv"
test_file = f"D:/RAG/XSS_attacks/dataset/XSS_dataset_testing_{test_size}.csv"

# **🔹 7. 直接用 UTF-8 儲存 CSV**
train_df.to_csv(train_file, index=False, encoding="utf-8", errors="replace")
test_df.to_csv(test_file, index=False, encoding="utf-8", errors="replace")

print(f"✅ 訓練集已保存至: {train_file} (共 {train_size} 筆，包括合法和惡意)")
print(f"✅ 測試集已保存至: {test_file} (共 {test_size} 筆，已確保 UTF-8)")

# **🔹 8. 確保 JSON 目錄存在**
json_dir = "D:/RAG/XSS_attacks/dataset/json"
os.makedirs(json_dir, exist_ok=True)  # 如果資料夾不存在則創建

# **🔹 9. 設定 JSON 文件輸出路徑**
json_file_path = os.path.join(json_dir, f"xss_dataset_training_{train_size}.json")

# **🔹 10. 轉換為 JSON 格式**
data = train_df.to_dict(orient="records")

# **🔹 11. 存成 JSON 檔案**
with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"✅ JSON 文件已成功保存至: {json_file_path}")

