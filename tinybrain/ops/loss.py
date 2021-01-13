import numpy as np
from ..core import Node
from ..ops import SoftMax 

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

class LogLoss(LossFunction):
    """
    对数损失
    """
    def compute(self):
        assert len(self.parents) == 1
         
        x = self.parents[0].value

        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent):
        """
        对数损失的导数为 -1/(1 + e^x)
        """
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
        return np.diag(diag.ravel())
    
class CrossEntropyWithSoftMax(LossFunction):
    """
    对第一个父节点（模型的线性输出部分）施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
    """
    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(-np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        # 交叉熵对logit部分的雅克比矩阵是logit与标签对应元素的差
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T
