import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# **1️⃣ 載入 CodeBERT 分類模型**
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print(f"✅ 成功載入模型: {model_name}")

# **2️⃣ 定義 XSS 分類函數**
def classify_xss_risk(user_input):
    """
    使用 CodeBERT 模型判斷 XSS Payload 風險。
    Args:
        user_input (str): 輸入的 XSS Payload。
    Returns:
        dict: 包含判斷結果和詳細信息的字典。
    """
    start_time = time.perf_counter()

    # **進行 Tokenization 並轉為 PyTorch 張量**
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # **前向傳播**
    with torch.no_grad():
        outputs = model(**inputs)

    # **檢查 logits 形狀**
    print(f"🔍 Logits Shape: {outputs.logits.shape}")  # ➜ 應該是 (1,2)

    # **確認模型輸出是二分類**
    if outputs.logits.shape[1] != 2:
        raise ValueError(f"模型 logits 維度錯誤: {outputs.logits.shape}，預期為 (1,2)")

    # **計算分類概率**
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

    # **判斷分類結果**
    predicted_label = np.argmax(probabilities)
    label_map = {0: "benign", 1: "malicious"}  # 0 = 合法 (benign), 1 = 惡意 (malicious)

    # **計算推論時間**
    inference_time_ms = (time.perf_counter() - start_time) * 1000  # 轉換為毫秒

    return {
        "input_payload": user_input,
        "classification": label_map[predicted_label],
        "probabilities": {label_map[0]: round(probabilities[0], 4), label_map[1]: round(probabilities[1], 4)},
        "inference_time_ms": round(inference_time_ms, 2)
    }

# **3️⃣ 測試 XSS 判斷**
test_payload = "<script>alert('XSS')</script>"
result = classify_xss_risk(test_payload)

# **4️⃣ 輸出結果**
print("\n🛠️ **判斷結果**")
print(f"🚀 輸入 Payload: {result['input_payload']}")
print(f"✅ 判斷結果: {result['classification']}")
print(f"📊 分類機率: {result['probabilities']}")
print(f"⏱️ 推論時間: {result['inference_time_ms']} ms")
