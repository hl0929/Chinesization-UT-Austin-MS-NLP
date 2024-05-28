import collections


class BigramLanguageModel:
    def __init__(self, sentences: list):
        """
        初始化Bigram模型
        :param sentences: 训练语料，类型为字符串列表
        """
        self.sentences = sentences
        self.word_counts = collections.Counter()
        self.bigram_counts = collections.defaultdict(int)
        self.unique_words = set()
        
        # 预处理数据：分词并合并所有句子
        words = ' '.join(sentences).split()
        for w1, w2 in zip(words[:-1], words[1:]):
            self.word_counts[w1] += 1
            self.word_counts[w2] += 1
            self.bigram_counts[(w1, w2)] += 1
            self.unique_words.update([w1, w2])
    
    def laplace_smooth(self, delta: float = 1.0):
        """
        拉普拉斯平滑
        :param delta: 平滑因子，默认为1.0
        """
        V = len(self.unique_words)  # 词汇表大小
        self.model = {}
        for w1 in self.unique_words:
            total_count_w1 = self.word_counts[w1] + delta*V
            self.model[w1] = {}
            for w2 in self.unique_words:
                count_w1w2 = self.bigram_counts.get((w1, w2), 0) + delta
                self.model[w1][w2] = count_w1w2 / total_count_w1
                
    def generate_text(self, start_word: str, length: int = 10) -> str:
        """
        生成文本
        :param start_word: 文本起始词
        :param length: 生成文本的长度
        :return: 生成的文本字符串
        """
        if start_word not in self.model:
            raise ValueError(f"Start word '{start_word}' not found in the model.")
        sentence = [start_word]
        current_word = start_word
        
        for _ in range(length):
            next_word_probs = self.model[current_word]
            next_word = max(next_word_probs, key=next_word_probs.get)
            sentence.append(next_word)
            current_word = next_word
            
        return ' '.join(sentence)

# 示例使用
corpus = [
    "ChatGPT is a powerful language model for multi task",
    "ChatGPT is a powerful language model",
    "ChatGPT can generate human-like text",
    "ChatGPT is trained using deep learning",
]

model = BigramLanguageModel(corpus)
model.laplace_smooth()
generated_text = model.generate_text('ChatGPT', 5)
print(generated_text)  # ChatGPT is a powerful language model

