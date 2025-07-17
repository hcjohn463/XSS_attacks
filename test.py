import os
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text  # ✅ 引入自動調整標籤套件
from tqdm import tqdm


# 模型清單（不要動）
model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "jackaduma/SecBERT",
    "microsoft/codebert-base",
    "roberta-base-openai-detector",
    "cssupport/mobilebert-sql-injection-detect",
]

# 對照表：圖例名稱
label_map = {
    "sentence-transformers/all-MiniLM-L6-v2": "Proposed Model",
    "jackaduma/SecBERT": "SecBERT",
    "microsoft/codebert-base": "CodeBERT",
    "roberta-base-openai-detector": "RoBERTa-detector",
    "cssupport/mobilebert-sql-injection-detect": "MobileBERT-SQL",
}

# 顏色
color_map = {
    "sentence-transformers/all-MiniLM-L6-v2": "#D62728",
    "jackaduma/SecBERT": "#1F77B4",
    "microsoft/codebert-base": "#2CA02C",
    "roberta-base-openai-detector": "#9467BD",
    "cssupport/mobilebert-sql-injection-detect": "#FF7F0E",
}

# 設定指標與範圍
metric = "Precision"
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

from tqdm import tqdm

# ⭐ 找出最高點（for 星號 + 特別標註）
max_idx = df[metric].idxmax()
x_max = df.loc[max_idx, "Malicious"]
y_max = df.loc[max_idx, metric]

# ⭐ 一般數值標籤（排除重複、相近值，保留頭尾）
prev_y = None
plotted_y_set = set()
first_idx = df.index[0]
last_idx = df.index[-1]

for i, (x, y) in enumerate(tqdm(zip(df["Malicious"], df[metric]), total=len(df), desc=f"標註 {label}")):
    force_label = (df.index[i] == first_idx or df.index[i] == last_idx)

    rounded_y = round(y, 1)
    rounded_y_max = round(y_max, 1)

    if not force_label:
        if x == x_max or rounded_y == rounded_y_max:
            continue
        if rounded_y in plotted_y_set:
            continue
        if prev_y is not None and abs(y - prev_y) < 1000:
            prev_y = y
            continue

    prev_y = y
    plotted_y_set.add(rounded_y)

    txt = plt.text(
        x, y,
        f"{y:.1f}%",
        fontsize=15,
        ha="center", va="bottom",
        color="black",
        bbox=dict(facecolor=color, edgecolor=color, alpha=0.3, boxstyle="round,pad=0.3")
    )
    all_texts.append(txt)

# ⭐ 特別標註最高點（星星 + 錯開的標籤文字）
# ➤ 星星在原地、文字錯開
x_offset = 0.25 * model_names.index(model_name)  # ← 可依需要調整間距
plt.scatter(x_max, y_max, color=color, edgecolors='black', zorder=5, s=90, marker='*')

star_text = plt.text(
    x_max + x_offset, y_max + 1.8,
    f"★ {y_max:.1f}% at {x_max}",
    fontsize=15,
    ha="center", va="bottom",
    color="black",
    bbox=dict(facecolor=color, edgecolor=color, alpha=0.5, boxstyle="round,pad=0.3")
)
all_texts.append(star_text)


# ✅ 自動調整文字位置，避免重疊
from adjustText import adjust_text
adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

# 圖設定
plt.title(f"Impact of Illegal Samples in Database: {metric}", fontsize=16)
plt.xlabel("Number of Illegal Samples in Database", fontsize=16)
plt.ylabel(f"{metric} (%)", fontsize=16)
plt.legend(fontsize=15)
plt.grid(True)
plt.xlim(start, end)
plt.tight_layout()
# 改 X/Y 軸的刻度數字大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


# 輸出
output_path = os.path.join(base_path, f"{metric}_comparison_{start}_{end}_newww.png")
plt.savefig(output_path, dpi=300)
plt.show()
