import jieba


sentence1 = "这个电影好，想再看这个电影。"
sentence2 = "这个电影不好，不想再看这个电影了。"

sentence1_tokenized = jieba.lcut(sentence1)
sentence2_tokenized = jieba.lcut(sentence2)
print("句子一分词结果:", sentence1_tokenized)
print("句子二分词结果:", sentence2_tokenized)


vocabulary = []
for word in sentence1_tokenized + sentence2_tokenized:
    if word not in vocabulary:
        vocabulary.append(word)
print("词表:", vocabulary)


sentence1_feature = [0] * len(vocabulary)
for word in sentence1_tokenized:
    index = vocabulary.index(word)
    sentence1_feature[index] += 1
print("句子一的特征向量(bow):", sentence1_feature)

sentence2_feature = [0] * len(vocabulary)
for word in sentence2_tokenized:
    index = vocabulary.index(word)
    sentence2_feature[index] += 1
print("句子一的特征向量(bow):", sentence2_feature)