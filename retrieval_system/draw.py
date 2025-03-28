import pandas as pd
import matplotlib.pyplot as plt

# 🔹 1. 指標選擇：Accuracy、Precision、Recall
metric = "Accuracy"  # 可改成 "Precision" 或 "Recall"

# 🔹 2. 讀取資料
# model_name = "microsoft/codebert-base"
# model_name = "jackaduma/SecBERT"
# model_name = "cssupport/mobilebert-sql-injection-detect"
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model_name = "roberta-base-openai-detector"
model_name = "BAAI/bge-small-en"

model_filename = model_name.replace('-', '_').replace('/', '_')
total_samples = 200
start = 0
end = 200

csv_path = rf"D:\RAG\XSS_attacks\result\retrieval\{total_samples}\XSS_summary_results_{model_filename}.csv"
df = pd.read_csv(csv_path)

# 🔹 3. 繪製
plt.figure(figsize=(15, 6))
plt.plot(df["Malicious"], df[metric], label=f"{model_name}", color="orange", marker = "o", markersize=2, linestyle="-", linewidth=2)


# 🔹 4. 添加標籤（✅ 每一個點都顯示）
previous = None
for x, y in zip(df["Malicious"], df[metric]):
    if previous is None or abs(y - previous) >= 1.0:
        plt.text(x, y + 1.4, f"{y:.1f}%", fontsize=8, ha="center", va="bottom",
                 bbox=dict(facecolor='white', alpha=0.6))
        previous = y


# 🔹 5. 標題與格式
plt.xlabel("Number of Malicious Samples in Training")
plt.ylabel(f"{metric} (%)")
plt.title(f"{metric} vs. Malicious Samples in Training ({model_name})")
plt.grid(True)
plt.legend()
plt.xlim(start, end)



# 🔹 6. 儲存
output_path = rf"D:\RAG\XSS_attacks\result\retrieval\{total_samples}\{metric}_vs_Malicious_{start}_{end}_{model_filename}.png"
plt.savefig(output_path, dpi=300)
print(f"✅ 圖片已儲存至 {output_path}")

# 🔹 7. 顯示
plt.show()
