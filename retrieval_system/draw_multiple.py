import os
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

# 模型清單
model_names = [
    "microsoft/codebert-base",
    "jackaduma/SecBERT",
    "cssupport/mobilebert-sql-injection-detect",
    "sentence-transformers/all-MiniLM-L6-v2",
    "roberta-base-openai-detector",
    "BAAI/bge-small-en"
]

metric = "Accuracy"
total_samples = 200
start, end = 0, 200
base_path = f"D:/RAG/XSS_attacks/result/retrieval/{total_samples}"

plt.figure(figsize=(18.75, 10))
texts = []

for model_name in model_names:
    filename = model_name.replace('-', '_').replace('/', '_')
    csv_path = os.path.join(base_path, f"XSS_summary_results_{filename}.csv")
    if not os.path.exists(csv_path):
        print(f"❌ 找不到檔案：{csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df = df[df["Malicious"].between(start, end)]

    line, = plt.plot(df["Malicious"], df[metric], label=model_name, marker='o', markersize=1)
    color = line.get_color()

    previous = None
    for x, y in zip(df["Malicious"], df[metric]):
        if previous is None or abs(y - previous) >= 5.0:
            txt = plt.text(x, y, f"{y:.1f}%", fontsize=8, ha="center", va="bottom",
                           bbox=dict(facecolor=color, edgecolor=color, alpha=0.3, boxstyle="round,pad=0.3"),
                           color="black")
            texts.append(txt)
            previous = y

    # 找到最高點
    max_idx = df[metric].idxmax()
    x_max = df.loc[max_idx, "Malicious"]
    y_max = df.loc[max_idx, metric]

    # 畫點 + 加上特殊標籤
    star = plt.scatter(x_max, y_max, color=color, edgecolors='black', zorder=5, s=80, marker='*')  # 星星標註
    txt = plt.text(
        x_max, y_max + 1.5, 
        f"★ {y_max:.1f}% at {x_max}", 
        fontsize=9, 
        ha="center", 
        va="bottom",
        color="black",
        bbox=dict(facecolor=color, edgecolor=color, alpha=0.5, boxstyle="round,pad=0.3")
    )
texts.append(txt)

# 美化與調整
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

plt.title(f"{metric} vs. Malicious Samples in Training (Multiple Models) {start}~{end}")
plt.xlabel("Number of Malicious Samples in Training")
plt.ylabel(f"{metric} (%)")
plt.legend()
plt.grid(True)
plt.xlim(start, end)
plt.tight_layout()

output_path = os.path.join(base_path, f"{metric}_comparison_{start}_{end}.png")
plt.savefig(output_path, dpi=300)
print(f"✅ 圖片已儲存：{output_path}")
plt.show()
