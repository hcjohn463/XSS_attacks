from transformers import AutoTokenizer

text = "<script>alert('XSS')</script>"

# BERT tokenizer
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
print("BERT tokens:", bert_tok.tokenize(text))

# Sentence-BERT tokenizer（例如：all-MiniLM-L6-v2）
sbert_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("SBERT tokens:", sbert_tok.tokenize(text))
