
class Node(object):
    def __init__(self, data, left_child=None, right_child=None):
        self.data = data
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        return str(self.data)


class Tree(object):
    def __init__(self, root):
        self.root = root


    def add(self, elem):
        node = Node(elem)
        if self.root == None:
            self.root = node
        else:
            queue = []
            queue.append(self.root)
            while queue:
                cur = queue.pop(0)
                if cur.left_child == None:
                    cur.left_child = node
                    return
                elif cur.right_child == None:
                    cur.right_child = node
                    return
                else:
                    queue.append(cur.right_child)
                    queue.append(cur.left_child)




def pre_order_search(tree):
    pass


def mid_order_search(tree):
    pass


def last_order_search(tree):
    pass


if __name__ == '__main__':
    node1 = Node(15)
    node2 = Node(7)
    node3 = Node(20, node1, node2)
    node4 = Node(9)
    base = Node(3, node4, node3)
    print(base.left_child)