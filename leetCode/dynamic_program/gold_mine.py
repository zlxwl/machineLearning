from typing import List
import copy

def solution(worker: int) -> int:
    # 错误2：数组越界的根本原因，金矿数目可以从 0-4 分别代表5个不同的金矿。但是在处理工人数量时没有将 工人数量为0考虑进去，仅仅将1-10人映射为数组0，9，少了一个状态
    # 解决方法： 初始化数组时注意工人数量这一个维度11而非10。
    # 错误2：用来存储上一个阶段挖矿情况的存储器，python的列表在进行赋值的时候并没有在内存中开辟一个新的空间，而是仅仅将两个变量指向同一个内存空间。a = b,
    # 解决方法：a = b[:] 替换在java和c++中存在的情况。
    g = [400, 500, 200, 300, 350]
    p = [5, 5, 3, 4, 3]

    w = worker
    n = len(g)

    # record = [0] * w
    result = [0] * (w+1)

    for i in range(0, w+1):
        if i < p[0]:
            result[i] = 0
        else:
            result[i] = g[0]
    record = result[:]
    print(result)

    for j in range(0, n):
        for i in range(0, w+1):
            if i < p[j]:
                result[i] = record[i]
            else:
                result[i] = max(record[i-p[j]]+g[j], record[i])
        print(result)
        record = result[:]

    return result[-1]


if __name__ == "__main__":
    print(solution(10))
