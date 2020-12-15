import numpy as np
from ..core import Node

def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
        filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled

class Operator(Node):
    """
    定义操作抽象类节点
    继承Node，并实现其compute以及get_jacobi两个方法
    """
    pass

class Add(Operator):
    """
    多个（不限于2个）父节点的矩阵加法
    """
    def compute(self):
        # assert len(self.parents) == 2 and self.parents[0].shape() == self.parents[1].shape()
        self.value = np.mat(np.zeros(self.parents[0].shape()))

        for parent in self.parents:
            self.value += parent.value
    
    def get_jacobi(self, parent):
        return np.mat(np.eye(self.dimension())) # 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵

class MatMul(Operator):
    """
    矩阵乘法（支持2个矩阵相乘）
    """
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value # 矩阵乘法
    
    def get_jacobi(self, parent):
        """
        TODO: 是真滴复杂~~~
        将矩阵乘法视作映射，求映射对参与计算的矩阵的雅克比矩阵。
        """
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(
                self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(
                parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]

class Step(Operator):
    
    def compute(self):
        """
        阶跃函数，当矩阵中值大于等于0.0时为1.0，否则为0.0
        TODO: 为什么是parents[0]
        """
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 1.0, 0.0))
    
    def get_jacobi(self, parent):
        """
        TODO: 没搞懂
        """
        np.mat(np.eye(self.dimension()))
        return np.zeros(np.where(self.parents[0].value.A1 >= 0.0, 0.0, -1.0))


class ScalarMultiply(Operator):
    """
    用标量（1x1矩阵）数乘一个矩阵
    TODO
    """

    def compute(self):
        assert self.parents[0].shape() == (1, 1)  # 第一个父节点是标量
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        assert parent in self.parents

        if parent is self.parents[0]:
            return self.parents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.parents[1].dimension())) * self.parents[0].value[0, 0]

class Multiply(Operator):
    """
    两个父节点的值是相同形状的矩阵，将它们对应位置的值相乘
    TODO
    """

    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)