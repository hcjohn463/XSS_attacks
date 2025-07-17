import os
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text  # ✅ 引入自動調整標籤套件

# 模型清單（不要動）
model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    # "jackaduma/SecBERT",
    # "microsoft/codebert-base",
    # "roberta-base-openai-detector",
    # "cssupport/mobilebert-sql-injection-detect",
]

# 對照表：圖例名稱
label_map = {
    "sentence-transformers/all-MiniLM-L6-v2": "Proposed Model",
    # "jackaduma/SecBERT": "SecBERT",
    # "microsoft/codebert-base": "CodeBERT",
    # "roberta-base-openai-detector": "RoBERTa-detector",
    # "cssupport/mobilebert-sql-injection-detect": "MobileBERT-SQL",
}

# 顏色
color_map = {
    "sentence-transformers/all-MiniLM-L6-v2": "#D62728",
    # "jackaduma/SecBERT": "#1F77B4",
    # "microsoft/codebert-base": "#2CA02C",
    # "roberta-base-openai-detector": "#9467BD",
    # "cssupport/mobilebert-sql-injection-detect": "#FF7F0E",
}

# 設定指標與範圍
metric = "Accuracy"
start, end = 0, 200
base_path = "D:/RAG/XSS_attacks/result/retrieval/200"

plt.figure(figsize=(18, 10))
all_texts = []  # ✅ collect all text objects to adjust later

for model_name in model_names:
    filename = model_name.replace('-', '_').replace('/', '_')
    csv_path = os.path.join(base_path, f"XSS_summary_results_{filename}.csv")
    if not os.path.exists(csv_path):
        print(f"❌ 找不到檔案：{csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df = df[df["Malicious"].between(start, end)]

    label = label_map[model_name]
    color = color_map[model_name]
    is_ours = model_name == "sentence-transformers/all-MiniLM-L6-v2"

    # 畫線
    plt.plot(
        df["Malicious"], df[metric],
        label=label,
        color=color,
        linewidth=2.8 if is_ours else 1.5,
        marker='o',
        markersize=4 if is_ours else 2,
        zorder=3 if is_ours else 2
    )

    # ⭐ 標註最高點
    max_idx = df[metric].idxmax()
    x_max = df.loc[max_idx, "Malicious"]
    y_max = df.loc[max_idx, metric]

    prev_y = None
    for i, (x, y) in enumerate(zip(df["Malicious"], df[metric])):
        if x == x_max:
            continue  # 跳過最高點，額外標註
        if prev_y is not None and abs(y - prev_y) < 0.5:  # ✅ 小於2%的變化就不標
            prev_y = y
            continue
        prev_y = y
        txt = plt.text(
            x, y,
            f"{y:.1f}%",
            fontsize=9,
            ha="center", va="bottom",
            color="black",
            bbox=dict(facecolor=color, edgecolor=color, alpha=0.3, boxstyle="round,pad=0.3")
        )
        all_texts.append(txt)

    # ⭐️ 特別標註最高點（加星星＋額外框）
    plt.scatter(x_max, y_max, color=color, edgecolors='black', zorder=5, s=90, marker='*')
    star_text = plt.text(
        x_max, y_max + 1.8,
        f"★ {y_max:.1f}% at {x_max}",
        fontsize=9,
        ha="center", va="bottom",
        color="black",
        bbox=dict(facecolor=color, edgecolor=color, alpha=0.5, boxstyle="round,pad=0.3")
    )
    all_texts.append(star_text)

# ✅ 自動調整文字位置，避免重疊
from adjustText import adjust_text
adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

# 圖設定
plt.title(f"Impact of Illegal Samples in Database: {metric}")
plt.xlabel("Number of Illegal Samples in Database")
plt.ylabel(f"{metric} (%)")
plt.legend()
plt.grid(True)
plt.xlim(start, end)
plt.tight_layout()

# 輸出
output_path = os.path.join(base_path, f"{metric}_comparison_{start}_{end}_only.png")
plt.savefig(output_path, dpi=300)
plt.show()
