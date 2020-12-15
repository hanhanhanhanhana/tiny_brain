import numpy as np
from ..core import Node

class LossFunction(Node):
    """
    Loss节点的抽象类
    """
    pass

class PerceptionLoss(LossFunction):
    """
    感知机损失，输入为正时为0，负时为输入的相反数
    """
    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi(self, parent):
        """
        雅克比矩阵为对角阵，每个对角线元素对应一个父节点元素。若父节点元素大于0，则
        相应对角线元素（偏导数）为0，否则为-1。
        """
        diag = np.where(parent.value >= 0.0, 0.0, -1.0)
        return np.diag(diag.ravel())