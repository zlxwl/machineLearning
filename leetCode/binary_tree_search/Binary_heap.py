

def up_adjust(array):
    child_index = len(array) - 1
    parent_index = int(child_index / 2)
    temp = array[child_index]
    while child_index > 0 and temp < array[parent_index]:
        # array[child_index] = temp
        array[child_index] = array[parent_index]
        child_index = parent_index
        parent_index = int(parent_index / 2)
    array[child_index] = temp
    return array


def down_adjust(array, parent_index):
    # 调整二叉堆，将所有非叶子节点下沉到合适位置，
    # 1.包括遍历所有非叶子节点。
    # 2.将叶子节点作为root，依次下沉，保证左右孩子不越界。
    # 3.定位到右孩子，满足parent>child
    # 则进行交换。
    # parent_index = 0
    child_index = 2*parent_index + 1
    temp = array[parent_index]
    #边界条件检测
    while child_index < len(array):
        # # 上浮与下沉不同的地方在于有两个子节点，且需要定位到右子节点，并且左右子节点均要保证数组不越界。
        if child_index + 1 < len(array) and array[child_index+1] < array[child_index]:
            child_index += 1
        if temp <= array[child_index]:
            break
        array[parent_index] = array[child_index]
        parent_index = child_index
        child_index = 2*child_index + 1
        # 方法2逐个判断。 实际是比较parent 和 left and right 之间的大小。 少了一个if条件 定位到右子节点
        # if child_index + 1 < len(array) and temp > array[child_index] and array[child_index] > array[child_index+1]:
        #     array[parent_index] = array[child_index]
        #     parent_index = child_index
        #     child_index = 2 * child_index + 1
        # elif child_index + 1 < len(array) and temp > array[child_index+1] and array[child_index] < array[child_index+1]:
        #     child_index += 1
        #     array[parent_index] = array[child_index]
        #     parent_index = child_index
        #     child_index = 2 * child_index + 1
        # elif temp <= array[child_index]:
        #     break
    array[parent_index] = temp
    return array


def build_heap(array):
    for i in range(int((len(array)-2)/2), -1, -1):
        down_adjust(array, i)


# 使用list代替数组模拟堆时可以不用在加入队列时进行数组复制扩容。list本身就是由数组扩容实现的。
from typing import List
# 优先队列
# 入队操作减少数组扩容部分，在末尾加入
def en_queue(array :List, value):
    return array.append(value)

# 出队操作，返回最大值，将数组末尾处数字转移到堆顶，并进行下沉。
def de_queue(array: List):
    # 创建堆
    build_heap(array)
    head = array[0]
    array[0] = array[-1]
    array.pop()
    return head


if __name__ == '__main__':
    # array = [1, 3, 2, 6, 5, 7, 8, 9, 10, 0]
    # print(up_adjust(array))

    # array = [7, 1, 3, 10, 5, 2, 8, 9, 6]
    # build_heap(array)
    # print(array)
    # print(array.pop())
    # print(array[len(array)-1])

    array = []
    en_queue(array, 3)
    en_queue(array, 5)
    en_queue(array, 10)
    en_queue(array, 2)
    en_queue(array, 7)
    print(array)
    a = de_queue(array)
    print(array)
    print(a)
    b = de_queue(array)
    print(array)
    print(b)