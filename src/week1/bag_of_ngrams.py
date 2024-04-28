import jieba


sentence1 = "这个电影好，想再看这个电影。"
sentence2 = "这个电影不好，不想再看这个电影了。"

sentence1_tokenized = jieba.lcut(sentence1)
sentence2_tokenized = jieba.lcut(sentence2)
print("句子一分词结果:", sentence1_tokenized)
print("句子二分词结果:", sentence2_tokenized)


vocabulary = []
tokenized = sentence1_tokenized + sentence2_tokenized
for two_gram in zip(tokenized, tokenized[1:]):
    two_gram_str = "".join(two_gram)
    if two_gram_str not in vocabulary:
        vocabulary.append(two_gram_str)
print("N-gram 词表:", vocabulary)

sentence1_feature = [0] * len(vocabulary)
for i in range(len(sentence1_tokenized) - 1):
    two_gram_str = "".join(sentence1_tokenized[i:i + 2])
    index = vocabulary.index(two_gram_str)
    sentence1_feature[index] += 1
print("句子一的特征向量(2-gram):", sentence1_feature)

sentence2_feature = [0] * len(vocabulary)
for i in range(len(sentence2_tokenized) - 1):
    two_gram_str = "".join(sentence2_tokenized[i:i + 2])
    index = vocabulary.index(two_gram_str)
    sentence2_feature[index] += 1
print("句子二的特征向量(2-gram):", sentence2_feature)