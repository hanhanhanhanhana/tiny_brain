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
        self.graph.clear_jacobi

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


