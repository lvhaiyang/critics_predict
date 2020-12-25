#encoding=utf8

from threading import main_thread
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os

class CriticsPredict:

    def __init__(self):
        # 训练完成的模型
        self.release_model = None
        #一个映射单词到整数索引的词典
        self.word_index = None
        # train_data      训练集(验证集)
        self.train_data = None
        # train_labels    训练集(验证集)标签
        self.train_labels = None
        # test_data       测试集
        self.test_data = None
        # test_labels     测试集标签
        self.test_labels = None
        self.load_data()

    def load_data(self):
        print("下载 IMDB 数据集")
        imdb = keras.datasets.imdb
        # train_data      训练集(验证集)
        # train_labels    训练集(验证集)标签
        # test_data       测试集
        # test_labels     测试集标签
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

        print("训练集数量: {}, 训练集标签数量: {}".format(len(train_data), len(train_labels)))
        print("评论文本被转换为整数值，其中每个整数代表词典中的一个单词。首条评论是这样的：")
        print(train_data[0])

        # 一个映射单词到整数索引的词典
        word_index = imdb.get_word_index()
        # print(word_index)
        # 保留第一个索引
        word_index = {k:(v+3) for k,v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2  # unknown
        word_index["<UNUSED>"] = 3

        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        # print(reverse_word_index)

        print("首条评论的文本：")
        print(' '.join([reverse_word_index.get(i, '?') for i in train_data[0]]))

        #一个映射单词到整数索引的词典
        self.word_index = word_index 
        # train_data      训练集(验证集)
        self.train_data = train_data
        # train_labels    训练集(验证集)标签
        self.train_labels = train_labels
        # test_data       测试集
        self.test_data = test_data
        # test_labels     测试集标签
        self.test_labels = test_labels

    def create_model(self):

        print("开始准备数据: 把评论填充相等的长度")
        # pad_sequences函数参数说明
        # train_data                  序列列表
        # value=word_index["<PAD>"]   填充值
        # padding='post',             在每个序列之后填充
        # maxlen=256                  序列最大值

        self.train_data = keras.preprocessing.sequence.pad_sequences(self.train_data,
                                                                value=self.word_index["<PAD>"],
                                                                padding='post',
                                                                maxlen=256)

        self.test_data = keras.preprocessing.sequence.pad_sequences(self.test_data,
                                                            value=self.word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)
        
        print("填充长度后的评论:")
        print(self.train_data[0])

        print("开始建模")
        # 输入形状是用于电影评论的词汇数目（10,000 词）
        # tf.keras 是用于构建和训练深度学习模型的 TensorFlow 高阶 API。利用此 API，可实现快速原型设计、先进的研究和生产，它具有以下三大优势：
        # 方便用户使用
        # Keras 具有针对常见用例做出优化的简单而一致的界面。它可针对用户错误提供切实可行的清晰反馈。
        # 模块化和可组合
        # 将可配置的构造块组合在一起就可以构建 Keras 模型，并且几乎不受限制。
        # 易于扩展
        # 可以编写自定义构造块，表达新的研究创意；并且可以创建新层、指标、损失函数并开发先进的模型。
        vocab_size = 10000
        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, 16))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.summary()

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        # 创建一个验证集
        # 在训练时，我们想要检查模型在未见过的数据上的准确率（accuracy）。
        # 通过从原始训练数据中分离 10,000 个样本来创建一个验证集。
        # （为什么现在不使用测试集？我们的目标是只使用训练数据来开发和调整模型，然后只使用一次测试数据来评估准确率（accuracy））。
        x_val = self.train_data[:10000]
        partial_x_train = self.train_data[10000:]

        y_val = self.train_labels[:10000]
        partial_y_train = self.train_labels[10000:]

        print("训练模型")
        model.fit(partial_x_train, #训练集
                    partial_y_train, #训练集标签
                    epochs=40, #批次
                    batch_size=512, #每个批次的数据量
                    validation_data=(x_val, y_val), #验证集和标签
                    verbose=1) #日志级别

        print("保存模型")
        if not os.path.exists('imdb_model'):
            os.mkdir('imdb_model')
        save_model_path = os.path.join('imdb_model', 'model{}'.format(int(time.time())))
        model.save(save_model_path)

        print("加载模型")
        my_model = keras.models.load_model(save_model_path)
        print("评估模型")
        results = my_model.evaluate(self.test_data,  self.test_labels, verbose=2)
        print(results)

    def load_model(self):
        print("加载已经训练好的模型") 
        save_model_path = os.path.join('imdb_model', 'release_model')
        self.release_model = keras.models.load_model(save_model_path)

    def predict_comment(self, comment):
        print("=" * 50)
        print("需要预测的评论内容：{}".format(comment))
        comment = comment.split(' ')
        predict_data = []
        for i in comment:
            if i in self.word_index.keys():
                predict_data.append(self.word_index[i])
            else:
                predict_data.append(2)

        predict_data = np.array([predict_data])
        print("将评论内容转换成模型输入:")
        print(predict_data)
        predict_data = keras.preprocessing.sequence.pad_sequences(predict_data,
                                                            value=self.word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)
        print("开始对评论进行预测:")
        output = self.release_model.predict(predict_data, verbose=3)
        for i in output:
            print("     模型输出结果:{}".format(i[0]))
            # sigmoid 激活函数范围 0-1， 所以大于0.5 位积极评论，小于0.5位消极评论
            if i[0] > 0.5:
                print("     这是一条积极的评论\n")
            else:
                print("     这是一条消极的评论\n")


if __name__ == "__main__":

    CP = CriticsPredict()
    # 训练新模型
    CP.create_model()

    # 加载训练完成的模型
    CP.load_model()
    # 使用模型预测评论
    # comment = "<START> this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert <UNK> is an amazing actor and now the same being director <UNK> father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for <UNK> and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also <UNK> to the two little boy's that played the <UNK> of norman and paul they were just brilliant children are often left out of the <UNK> list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all"
    # comment = 'This movie is a terrible waste of time'
    comment = 'The film is very good'
    CP.predict_comment(comment)

    comment = 'This movie is a terrible waste of time'
    CP.predict_comment(comment)