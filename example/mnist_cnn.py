import sys
sys.path.append('../')

import numpy as np
import struct
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import tinybrain as tb


# 读取标签数据集
with open('../data/train-labels.idx1-ubyte', 'rb') as lbpath:
    labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)

# 读取图片数据集
with open('../data/train-images.idx3-ubyte', 'rb') as imgpath:
    images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols) 

print(labels.shape)
print(images.shape)
print(labels.reshape(-1, 1).shape)

images = images[:100] / 255
labels = labels.astype(np.int)[:100]


# 将整数形式的标签转换成One-Hot编码
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(labels.reshape(-1, 1))

img_shape = (28, 28)

x = tb.core.Variable(dim=img_shape, init=False, trainable=False)

conv1 = tb.layer.conv([x], img_shape, 3, (5,5), "ReLU")

pooling1 = tb.layer.pooling(conv1, (3,3), (2,2))

conv2 = tb.layer.conv(pooling1, (14,14), 3, (3,3), "ReLU")

pooling2 = tb.layer.pooling(conv2, (3,3), (2,2)) # 输出为数组[(7,7), (7,7), (7,7)]

fc1 = tb.layer.fc(tb.ops.Concat(*pooling2), 147, 120, "ReLU") # 将 

output = tb.layer.fc(fc1, 120, 10, "None")

one_hot = tb.core.Variable(dim=(10, 1), init=False, trainable=False)

# 模型输出
predict = tb.ops.SoftMax(output)

# 交叉熵损失
loss = tb.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.005

# 构造Adam优化器，GradientDescent不行
optimizer = tb.optimizer.Adam(tb.core.default_graph, loss, learning_rate)

# 批大小为16
batch_size = 32

# 训练执行200个epoch
for epoch in range(50):

    # 批计数器清零
    batch_count = 0

    # 遍历训练集中的样本
    for i in range(len(images)):

        feature = np.mat(images[i]).reshape(img_shape)

        # 取第i个样本的One-Hot标签，3x1矩阵
        label = np.mat(one_hot_label[i]).T

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

    for i in range(len(images)):
        feature = np.mat(images[i]).reshape(img_shape)
        x.set_value(feature)

        predict.forward()
        pred.append(predict.value.A.ravel()) # .A代表将矩阵转化为array数组类型

    pred = np.array(pred)
    pred = np.argmax(pred,axis=1)
    acc = (pred == labels).astype(np.int).sum()/len(images)
    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, acc))

