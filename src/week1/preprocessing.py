import jieba
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('punkt')

# 中文分词和停用词过滤
stopwords = ['的', '了', '和', '是', '就', '都', '而', '及', '与', '或', '个', '也', '这']
text = "我喜欢看电影，尤其喜欢科幻电影和动作片。"
tokenized_list = jieba.lcut(text)
print("中文分词结果:", tokenized_list)

tokenized_remove_stopword_list = [word for word in tokenized_list if word not in stopwords]
print("中文停用词过滤后的分词结果:", tokenized_remove_stopword_list)


# 英文分词和词形还原
text = "cats ate running better"
words = word_tokenize(text)
print("英文分词结果:", words)

lemmatizer = WordNetLemmatizer()  # 词形还原器
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("词形还原结果:", lemmatized_words)

stemmer = PorterStemmer()         # 词干提取器
stemmed_words = [stemmer.stem(word) for word in words]
print("词干提取结果:", stemmed_words)
