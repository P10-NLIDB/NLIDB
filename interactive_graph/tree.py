class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.children_relation = dict()
        self._size = -1
        self._depth = -1
        self.parents = []


    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)
    
    def add_parent(self, parent):
        self.parents.extend(parent)

    def add_relation(self, child_node, relation):
        self.children_relation[child_node] = relation

    def size(self):
        if getattr(self, '_size') != -1:
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth') != -1:
            return self._depth
        count = 1
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

