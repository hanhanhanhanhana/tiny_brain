import numpy as np
import abc
from .graph import default_graph

class Node():
    """
    计算图节点基类
    parents: 构建图节点时的输入，如MatMul(list_x, list_y)，则parents则为tuple(list_x, list_y)
    """
    def __init__(self, *parents, **kargs):
        # 计算图对象，若无则默认为全局对象default_graph
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.gen_node_name(**kargs)

        self.parents = list(parents) #搭建图时，已知父节点，未知子节点
        self.children = []
        self.value = None # 本节点的值
        self.jacobi = None # 结果节点对本节点的雅可比矩阵

        # 搭建图时，将本节点添加到父节点的子节点列表中
        for parent in parents:
            parent.children.append(self)

        # 将本节点添加到计算图中
        self.graph.add_node(self)
    
    def get_parents(self):
        """
        得到本节点的父节点
        """
        return self.parents
    
    def get_children(self):
        """
        得到本节点的子节点
        """
        return self.children
    
    def gen_node_name(self, **kargs):
        """
        生成节点的名称，若用户不指定，则根据节点类型生成类似于MatMul:3的节点名，
        若用户指定了name_scope，则生成类似于Hidden/MatMul:3的节点名
        """
        self.name = kargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()))
        if self.graph.name_scope:
            self.name = '{}:{}'.format(self.graph.name_scope, self.name)
    
    def forward(self):
        """
        前向传播本节点的值，若父节点的值未被计算，则递归调用父节点的forward
        """
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.compute()
    
    @abc.abstractmethod
    def compute(self):
        """
        抽象方法，根据父节点的值计算本节点的值
        """
    
    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        抽象方法，计算本节点对某个父节点的雅可比矩阵
        """

    def backward(self, result):
        """
        反向传播，计算结果节点对本节点的雅可比矩阵，result是最终的结果节点。（一般情况下，最终结果result是一个标量-损失值，
        它对计算路径上游某个节点的雅可比矩阵是1×n的矩阵，n为该节点的维数，这个1×n的雅可比矩阵就是最终结果对该节点的梯度的转置）
        若本节点是最终的结果节点，那么只需要构造一个适当形状的单位矩阵作为雅克比矩阵即可；
        若不是，则首先构造一个适当形状的全零矩阵作为累加器，再遍历其全部子节点，
        若子节点value不为None，说明它在此计算路径上，递归调用子节点的backward，获得结果节点对它的雅克比矩阵，
        然后通过get_jacobi得到子节点对自己节点的雅克比矩阵，相乘累加，
        当子节点遍历完成后，累加器的值就是最终结果对本节点的雅克比矩阵。
        反向传播向父节点传递的二元组为：最终结果对自己的雅克比矩阵以及自己对父节点的雅可比矩阵。
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                self.jacobi = np.mat(np.zeros((result.dimension(), self.dimension())))
            for child in self.get_children():
                if child.value is not None:
                    self.jacobi += child.backward(result) * child.get_jacobi(self)
        return self.jacobi
                
    def clear_jacobi(self):
        """
        清空结果节点对本节点的雅可比矩阵
        """
        self.jacobi = None
    
    def dimension(self):
        """
        返回本节点的值展平成向量后的维数
        """
        return self.value.shape[0]*self.value.shape[1]

    def shape(self):
        """
        返回本节点的值作为矩阵的形状：（行数，列数）
        """
        return self.value.shape
    
    def reset_value(self, recursive=True):
        """
        重置本节点的值，并递归重置本节点的下游节点的值
        """
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()

class Variable(Node):
    """
    变量节点
    TODO: 并未实现基类的compute与get_jacobi
    """
    def __init__(self, dim, init=False, trainable=True, **kargs):
        """
        变量节点，无父节点，构造函数接受变量的形状，是否初始化以及是否参与训练的标识
        """
        Node.__init__(self, **kargs)
        self.dim = dim
        
        # 如果需要初始化，则以正态分布随机初始化变量的值
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))
        
        self.trainable = trainable
    
    def set_value(self, value):
        """
        为变量赋值
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim

        # 本节点的值被改变，重置下游节点的值
        self.reset_value()
        self.value = value





    



