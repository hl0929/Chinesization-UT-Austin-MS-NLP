from transformers import BertTokenizer

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 输入文本
text = "The quick brown fox jumps over the lazy dog."

# 分词
tokens = tokenizer.tokenize(text)
print(tokens)
# ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

# 编码
input_ids = tokenizer.encode(text, add_special_tokens=True)
print(input_ids)
# [101, 1996, 4248, 2829, 4419, 2169, 2058, 1996, 13971, 3899, 1012, 102]