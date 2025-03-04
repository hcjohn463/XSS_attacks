input_file = "D:/RAG/xss_attacks/dataset/XSS_dataset_testing_utf8.csv"
output_file = "D:/RAG/xss_attacks/dataset/XSS_dataset_testing_cleaned.csv"

# 轉換時忽略無法解碼的錯誤字元
with open(input_file, "r", encoding="Windows-1252", errors="ignore") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    f_out.write(f_in.read())

print(f"✅ 已成功將 CSV 轉為 UTF-8，存為 {output_file}")
