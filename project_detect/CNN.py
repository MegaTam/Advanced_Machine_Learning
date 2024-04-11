import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
# from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D


data = pd.read_csv("processed_data.csv", encoding='unicode_escape').head(10)

data1 = data.loc[:, "text"]        # content
data2 = data.loc[:, "generated"]   # label

data_new = pd.concat([data1, data2], axis=1)
# print(data_new)

# Seq_2_Seq
# 假设我们有以下文本和标签
texts = data1.astype(str)
labels = data2  # 1代表AI生成，0代表人类生成

# 对文本进行分词
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 对序列进行填充，使它们具有相同的长度
data = pad_sequences(sequences)

# 定义模型
model = Sequential()
model.add(Embedding(10000, 128, input_length=data.shape[1]))
model.add(Conv1D(32, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(data, labels, epochs=10, validation_split=0.2)
