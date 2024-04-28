from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


# 加载SST-2数据集
dataset = load_dataset("glue", "sst2")

# 获取训练集和验证集
train_data = dataset["train"]
valid_data = dataset["validation"]

# 初始化词袋模型
vectorizer = CountVectorizer()

# 将训练集文本转换为词袋特征表示
train_features = vectorizer.fit_transform(train_data["sentence"])

# 将验证集文本转换为词袋特征表示
valid_features = vectorizer.transform(valid_data["sentence"])

# 获取训练集标签
train_labels = train_data["label"]

# 初始化逻辑回归模型，并增加最大迭代次数
classifier = LogisticRegression(max_iter=1000)  # 增加最大迭代次数为1000

# 训练情感分类模型
classifier.fit(train_features, train_labels)

# 预测训练集和验证集标签
train_predictions = classifier.predict(train_features)
valid_predictions = classifier.predict(valid_features)

# 计算训练集和验证集准确率
train_accuracy = accuracy_score(train_labels, train_predictions)
valid_accuracy = accuracy_score(valid_data["label"], valid_predictions)

print("训练集准确率：", train_accuracy)
print("验证集准确率：", valid_accuracy)
