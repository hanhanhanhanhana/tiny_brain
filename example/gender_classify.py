import sys
sys.path.append('../')

import numpy as np
import tinybrain as tb

"""
制造训练样本。根据均值170，标准差5的正态分布采样300个男性身高，根据均值160，
标准差5的正态分布采样300个女性身高。根据均值120，标准差5的正态分布采样300个
男性体重，根据均值100，标准差5的正态分布采样300个女性体重。
构造300个1，作为男性标签，构造300个-1，作为女性标签。将数据组装成一个
600 x 3的numpy数组，前2列分别是身高、体重，最后一列是性别标签。
"""
male_height = np.random.normal(190, 5, 300)
female_height = np.random.normal(160, 5, 300)

male_weight = np.random.normal(80, 5, 300)
female_weight = np.random.normal(50, 5, 300)

male_label = [1] * 300
female_label = [-1] * 300

train_set = np.array([np.concatenate((male_height, female_height)),
                    np.concatenate((male_weight, female_weight)),
                    np.concatenate((male_label, female_label))]).T

np.random.shuffle(train_set)

# 构造计算图，输入向量为一个2×1的矩阵（代表），不需要初始化，不参与训练
x = tb.core.Variable(dim=(2,1), init=False, trainable=False)

# 类别标签，1男，-1女
label = tb.core.Variable(dim=(1,1), init=False, trainable=False)

# 权重向量，2×1的向量，需要初始化，参与训练
w = tb.core.Variable(dim=(1,2), init=True, trainable=True)

# 阈值，是一个1x1矩阵，需要初始化，参与训练
b = tb.core.Variable(dim=(1, 1), init=True, trainable=True)

# 预测
output = tb.ops.Add(tb.ops.MatMul(w, x), b)
pred = tb.ops.Step(output)

# 损失函数
loss = tb.ops.loss.PerceptionLoss(tb.ops.MatMul(label, output))

lr = 0.0001

for epoch in range(50):
    # 遍历所有训练集中的数据
    for i in range(len(train_set)):
        # 构造2x1特征矩阵
        features = np.mat(train_set[i, :-1]).T
        
        l = np.mat(train_set[i, -1])
        
        # 将特征赋给x节点，将标签赋给label节点
        x.set_value(features)
        label.set_value(l)

        # 在loss节点上执行前向传播，计算损失值
        loss.forward()

        # 在w和b节点上执行反向传播，计算损失值对它们的雅可比矩阵
        w.backward(loss)
        b.backward(loss)
        """
        用损失值对w和b的雅可比矩阵（梯度的转置）更新参数值。最终结果节点对
        变量节点的雅可比矩阵的形状都是1 x n。这个雅可比的转置是结果节点对
        变量节点的梯度。将梯度再reshape成变量矩阵的形状，对应位置上就是结
        果节点对变量元素的偏导数。将改变形状后的梯度乘上学习率，从当前变量
        值中减去，再赋值给变量节点，完成梯度下降更新。
        """
        w.set_value(w.value - lr * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - lr * b.jacobi.T.reshape(b.shape()))

        # 清除图中所有节点的雅可比矩阵
        tb.core.default_graph.clear_jacobi()

    # 每个epoch后评价模型的准确率
    pred_list = []

    for i in range(len(train_set)):
        features = np.mat(train_set[i,:-1]).T
        x.set_value(features)
        
        pred.forward()
        pred_list.append(pred.value[0, 0]) # pred为二维矩阵
    
    pred_list = np.array(pred_list) * 2 - 1  # 将1/0结果转化成1/-1结果，好与训练标签的约定一致
    
    acc = (train_set[:, -1] == pred_list).astype(np.int).sum() / len(train_set)

    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, acc))






        


