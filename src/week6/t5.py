from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载T5模型和分词器
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 输入文本
text = "It is reported that a major fire broke out in Victoria Harbor in Hong Kong on December 12, which injured 100 people and caused 10 billion yuan in damage"

# 将任务和输入文本结合
input_text = "summarize: " + text

# 编码输入文本
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 生成摘要
summary_ids = model.generate(input_ids, max_length=50, num_beams=2, early_stopping=True)

# 解码生成的摘要
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Summary:", summary)