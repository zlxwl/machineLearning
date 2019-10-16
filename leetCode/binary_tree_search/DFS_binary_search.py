from typing import List
from collections import deque

class Node(object):
    def __init__(self, data, left_child=None, right_child=None):
        self.data = data
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        return str(self.data)


# class Tree(object):
#     def __init__(self, root):
#         self.root = root
#
#
#     def add(self, elem):
#         node = Node(elem)
#         if self.root == None:
#             self.root = node
#         else:
#             queue = []
#             queue.append(self.root)
#             while queue:
#                 cur = queue.pop(0)
#                 if cur.left_child == None:
#                     cur.left_child = node
#                     return
#                 elif cur.right_child == None:
#                     cur.right_child = node
#                     return
#                 else:
#                     queue.append(cur.right_child)
#                     queue.append(cur.left_child)
def create(array: List):
    node = Node(data=None)
    if array is None or len(array) == 0:
        return None
    data = array.pop(0)
    if data != -1:
        node.data = data
        node.left_child = create(array)
        node.right_child = create(array)
    return node


def pre_order_search(tree):
    if tree is None:
        return
    print(tree.data)
    pre_order_search(tree.left_child)
    pre_order_search(tree.right_child)


def pre_order_search_with_stack(tree):
    stack = []
    tree_node = Node(tree.data, tree.left_child, tree.right_child)
    while stack or tree_node:
        while tree_node:
            print(tree.data)
            stack.append(tree_node.left_child)
            tree_node = tree_node.left_child
        if stack:
            tree_node = stack.pop()
            tree_node = tree_node.right_child


def mid_order_search(tree):
    if tree is None:
        return
    mid_order_search(tree.left_child)
    print(tree.data)
    mid_order_search(tree.right_child)


def last_order_search(tree):
    if tree is None:
        return
    last_order_search(tree.left_child)
    last_order_search(tree.right_child)
    print(tree.data)


def level_order_search(tree):
    search_queue = deque()
    # 正确表达节点的信息，通过节点对象的指针去获取左右节点。单纯通过获得data 的方式无法遍历打印
    search_queue.append(Node(tree.data, tree.left_child, tree.right_child))
    while search_queue:
        node = search_queue.popleft()
        print(node)
        if node.left_child:
            search_queue.append(node.left_child)
        if node.right_child:
            search_queue.append(node.right_child)


if __name__ == '__main__':
    # node1 = Node("A", "B", "C")
    # node2 = Node("B", None, "D")
    # node3 = Node("C", "E", "F")
    # node4 = Node("E", "G", None)
    # node5 = Node("F", "H", "I")
    # 通过列表的方式创建数

    tree_list = [3, 2, 9, -1, -1, 10, -1, -1, 8, -1, 4]
    tree = create(tree_list)
    # pre_order_search(tree)
    # mid_order_search(tree)
    # last_order_search(tree)

    # pre_order_search_with_stack(tree)
    level_order_search(tree)
    # search_queue = deque()
    # search_queue += search_queue + Node
    # print(tree)

    # node1 = Node(15)
    # node2 = Node(7)
    # node3 = Node(20, node1, node2)
    # node4 = Node(9)
    # base = Node(3, node4, node3)
