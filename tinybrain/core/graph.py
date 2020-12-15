class Graph:
    """
    计算图类
    """
    def __init__(self):
        self.nodes = [] # 计算图中的节点的列表
        self.name_scope = None
    
    def add_node(self, node):
        """
        添加节点
        """
        self.nodes.append(node)
    
    def clear_jacobi(self):
        """
        清楚图中所有节点的雅乐比矩阵
        """
        for node in self.nodes:
            node.clear_jacobi()
        
    def reset_value(self):
        """
        重置图中所有节点的值
        """
        for node in self.nodes:
            node.reset_value(False)
    
    def node_count(self):
        """
        得到图中节点的数量
        """
        return len(self.nodes)
    
    def draw(self, ax=None):
        """
        TODO
        """
        pass

# 全局默认计算图
default_graph = Graph()
