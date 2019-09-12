import random


# 这种实现方式需要复制数组，空间复杂度高
def quick_sort(array):
    if len(array) < 2:
        return array
    else:
        # pivot = array[0]
        # 改成随机选取
        pivot = random.choice(array)
        return quick_sort([a for a in array if a < pivot]) + [a for a in array if a == pivot] + quick_sort([a for a in array if a > pivot])


# 节省空间，交换元素替代数组复制，空间复杂度降为 O(n*logn)
def quick_sort_inplace_partition(array, left, right, pivot_index):

    def switch(array, pivot_index, right):
        temp = array[pivot_index]
        array[pivot_index] = array[right]
        array[right] = temp

    pivot_value = array[pivot_index]
    # 将pivot移动到数组末尾
    switch(array, pivot_index, right)
    # 扫描指针
    store_index = left
    for i in range(left, right):
        if array[i] <= pivot_value:
            switch(array, i, store_index)
            store_index += 1
    # 将pivot移动到排序后的store_index上，这个就是pivot的最终位置。即pivot有序
    switch(array, store_index, right)
    return store_index, array


def quick_sort_inplace(array, left, right):
    # 左右指针在递归调用时同时指向同一个元素时，递归调用终止。
    # 每次递归保证选中的pivot_index, 即mid_value = array[pivot_index]，所有pivot_index在该次调用结束后都有序，即中间位置。
    # 原先的那种写法，无法将pivot_index传递到递归函数中。 pivot_index应该是针对分治以后的小数组而言的。
    if left < right:
        pivot_index = left
        pivot_new_index, array = quick_sort_inplace_partition(array, left, right, pivot_index)
        quick_sort_inplace(array, left, pivot_new_index - 1)
        quick_sort_inplace(array, pivot_new_index + 1, right)



def quick_sort_inplace_v2(array, left, right):
    if left >= right:
        return

    pivot_value = array[left]
    low = left
    high = right

    # 双向指针扫描一次完成后，将pivot置入指定的位置。
    while low < high:
        while low < high and array[high] >= pivot_value:
            high -= 1
        array[low] = array[high]
        while low < high and array[low] < pivot_value:
            low += 1
        array[high] = array[low]
    array[low] = pivot_value
    quick_sort_inplace_v2(array, left, low-1)
    quick_sort_inplace_v2(array, low+1, right)


if __name__ == "__main__":
    array = [3, 7, 8, 5, 2, 1, 9, 5, 4]
    quick_sort_inplace(array, 0, len(array)-1)
    print(array)
