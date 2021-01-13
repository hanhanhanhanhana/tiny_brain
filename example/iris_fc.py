import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tinybrain as tb

# 读取鸢尾花数据集，去掉第一列Id
data = pd.read_csv("../data/Iris.csv").drop("Id", axis=1)

# 随机打乱样本顺序
data = data.sample(len(data), replace=False)

# 将字符串形式的类别标签转换成整数0，1，2
le = LabelEncoder()
number_label = le.fit_transform(data["Species"])

# 将整数形式的标签转换成One-Hot编码
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(number_label.reshape(-1, 1)) # 首先将number_label reshape为二维数组

# 特征列
features = data[['SepalLengthCm',
                 'SepalWidthCm',
                 'PetalLengthCm',
                 'PetalWidthCm']].values

# print(features.shape) (150,4)

x = tb.core.Variable(dim=(4,1), init=False, trainable=False)

hidden_1 = tb.layer.fc(x, 4, 10, "ReLU")

hidden_2 = tb.layer.fc(hidden_1, 10, 10, "ReLU")

output = tb.layer.fc(hidden_2, 10, 3, None)

one_hot = tb.core.Variable(dim=(3,1), init=False, trainable=False)

# 模型输出
predict = tb.ops.SoftMax(output)

# 交叉熵损失
loss = tb.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.01

# 构造Adam优化器，GradientDescent不行
optimizer = tb.optimizer.GradientDescent(tb.core.default_graph, loss, learning_rate)

# 批大小为16
batch_size = 16

# 训练执行200个epoch
for epoch in range(200):

    # 批计数器清零
    batch_count = 0

    # 遍历训练集中的样本
    for i in range(len(features)):

        # 取第i个样本，构造4x1矩阵对象
        feature = np.mat(features[i,:]).T

        # 取第i个样本的One-Hot标签，3x1矩阵
        label = np.mat(one_hot_label[i,:]).T

        x.set_value(feature)
        one_hot.set_value(label)

        # 调用优化器的one_step方法，执行一次前向传播和反向传播
        optimizer.one_step()
        
        # 批计数器加1
        batch_count += 1
        
        # 若批计数器大于等于批大小，则执行一次梯度下降更新，并清零计数器
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0
    
    pred = []

    for i in range(len(features)):
        feature = np.mat(features[i,:]).T
        x.set_value(feature)

        predict.forward()
        pred.append(predict.value.A.ravel()) # .A代表将矩阵转化为array数组类型

    pred = np.array(pred)
    pred = np.argmax(pred,axis=1)
    acc = (pred == number_label).astype(np.int).sum()/len(data)
    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, acc))

