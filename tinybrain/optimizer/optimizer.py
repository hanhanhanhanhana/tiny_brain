import numpy as np
import abc
from ..core import Graph
from ..core import Node, Variable


class Optimizer():
    """
    优化器基类
    """
    def __init__(self, graph, target, learning_rate=0.01):
        """
        优化器构造函数接受计算图对象，目标节点对象以及学习率
        """
        assert isinstance(graph, Graph) and isinstance(target, Node)
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        self.acc_gradient = dict()
        self.acc_no = 0 # 用来计数forward了多少次，一次一个样本，当acc_no等于bs时，重置

    def forward_backward(self):
        """
        前向传播计算结果节点的值并反向传播计算结果节点对各个节点的雅可比矩阵
        """

        # 清除图中所有节点的雅可比矩阵
        self.graph.clear_jacobi()

        # 前向传播计算结果节点
        self.target.forward()

        # 反向传播计算雅可比矩阵
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)

                # 最终结果（标量）对节点值的雅可比是一个行向量，其转置是梯度（列向量）
                # 这里将梯度reshape成与节点值相同的形状，好对节点值进行更新。
                # 若当前acc_no未达到指定bs，则梯度进行累加
                gradient = node.jacobi.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient
                    
    def one_step(self):
        """
        计算并累计样本的梯度
        """
        self.forward_backward()
        self.acc_no += 1
    
    def get_gradient(self, node):
        """
        返回梯度累加器的平均梯度
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no
    
    @abc.abstractmethod
    def _update(self):
        """
        抽象方法，执行具体的梯度更新算法，由子类实现
        """
    
    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):
        """
        TODO
        """
        for node, gradient in node_gradients_dict.items():
            if isinstance(node, Node):
                pass
            else:
                target_node = get_node_from_graph(node)
                assert target_node is not None
                assert self.acc_gradient[target_node].shape == gradient.shape
                if summarize:
                    self.acc_gradient[target_node] += gradient
                else:
                    self.acc_gradient[target_node] = gradient

        if summarize:
            self.acc_no += acc_no
        else:
            if acc_no is None:
                # 传入的是平均梯度, 强制让acc_no变为1，避免梯度更新时重复平均
                self.acc_no = 1
            else:
                self.acc_no = acc_no

    def update(self, var_gradients=None):

        if var_gradients is not None:
            self.apply_gradients(var_gradients)

        # 执行更新
        self._update()

        # 清除累加梯度
        self.acc_gradient.clear()
        self.acc_no = 0
    

        
class GradientDescent(Optimizer):
    """
    梯度下降优化器
    """
    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
    
    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # 得到该节点在当前batch的梯度
                gradient = self.get_gradient(node)
                # 用朴素梯度下降法更新变量节点的值
                node.set_value(node.value - self.learning_rate * gradient)

class Adam(Optimizer):
    """
    Adam优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99):

        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

        # 历史梯度衰减系数
        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1

        # 历史梯度各分量平方衰减系数
        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2

        # 历史梯度累积
        self.v = dict()

        # 历史梯度各分量平方累积
        self.s = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    # 梯度累积
                    self.v[node] = self.beta_1 * self.v[node] + \
                        (1 - self.beta_1) * gradient

                    # 各分量平方累积
                    self.s[node] = self.beta_2 * self.s[node] + \
                        (1 - self.beta_2) * np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.learning_rate *
                               self.v[node] / np.sqrt(self.s[node] + 1e-10))


