# coding=utf-8
"""
@version: 1.0
@author: William Zhong
@license: Best
@contact: 625015751@qq.com
"""
import math
from queue import Queue
# import random as ra
import pylab as pl
import numpy as np
import numpy.random as ra
from numpy import random
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
# from AggregationMethod.AP_aggragation import getText, cut_service
from sklearn.manifold import TSNE


def generate_random_number():
    return random.random(size=(100, 2))


def cal_euclidean_distance(dist1, dist2):
    distance = math.sqrt(sum((dist1 - dist2) * (dist1 - dist2).T))
    return distance


def DBSCAN_algorithm(data, e, minPts):
    m, n = data.shape
    data_dict = dict((x, y) for x, y in enumerate(data))  # 原始数据矩阵。
    distance_matix = np.zeros((m, m))
    # 距离矩阵
    for i in range(m):
        # print(i)
        for j in range(i + 1, m):
            distance_matix[i][j] = cal_euclidean_distance(data_dict[i], data_dict[j])
            distance_matix[j][i] = distance_matix[i][j]

    # 寻找领域集合
    neiborhood_dict = {}  # 领域
    core_points = []  # 核心对象
    for i in range(m):
        neiborhood_dict[i] = []
        for j in range(m):
            # if distance_matix[i][j] < e and distance_matix[i][j] != 0:
            if distance_matix[i][j] <= e :
                neiborhood_dict[i].append(j)

    # 确定核心对象
    for x, y in neiborhood_dict.items():
        if len(y) >= minPts:
            core_points.append(x)
    # cal_list_for = []
    # for c in core_points:
    #     Nq = [x for (x,y) in data_dict.items() if cal_euclidean_distance(y,data_dict[c]) <=e ]
    #     cal_list_for.extend(Nq)
    # print(len(list(set(cal_list_for))))
    # print(list(set([ cal_list.extend(neiborhood_dict[x]) for x in core_points])))
    if len(core_points) ==0:
        raise  'please set e and minpts correctly!'
    cluster_index = 0
    cluster_labels = {}
    unvisit_samples = list(neiborhood_dict.keys())
    all_samples = unvisit_samples[:]
    while len(core_points):
        unvisit_samples_old = unvisit_samples[:]
        core_index = ra.randint(0, len(core_points))
        core_queue = Queue()
        # print('core_index',core_index)
        # print('core_point',core_points[core_index])
        # print('core_point_list',core_points)
        core_queue.put(core_points[core_index])
        unvisit_samples.remove(core_points[core_index])
        index = 0
        while not core_queue.empty():
            sample = core_queue.get()
            # print(sample)
            if len(neiborhood_dict[sample]) >= minPts:
                delta = list(set(neiborhood_dict[sample]) & (set(unvisit_samples)))
                for element in delta:
                    core_queue.put(element)
                unvisit_samples = list(set(unvisit_samples) - set(delta))
            index+=1
        cluster_index += 1
        cluster_labels[cluster_index] = list(set(unvisit_samples_old) - set(unvisit_samples))
        core_points = list(set(core_points) - set(cluster_labels[cluster_index]))
        #错误
        # core_points = list(set(core_points) -set(cluster_labels.keys()))
        print('cluster_after',cluster_index,core_points)
    list_classified = []
    for x, y in cluster_labels.items():
        list_classified.extend(y)

    #记录未被分类的坐标点
    cluster_labels[0] = list(set(all_samples)- set(list_classified))
    print('%s: noise point' % (len(cluster_labels[0])))
    #class[0] is noise point
    return cluster_labels,data


def draw_graphy(cluster_labels,random_number):
    result = random_number
    # result = TSNE().fit_transform(random_number)
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for (_,y), col in zip(cluster_labels.items(), colors):
        class_members = y
        # 绘制每个类别下的坐标
        plt.plot(result[class_members, 0], result[class_members, 1], col + '.')
    for clusters, members in cluster_labels.items():
        for member in members:
            if clusters == 0:
                plt.annotate('noi', xy=(result[member, 0], result[member, 1]))
            else:
                plt.annotate(clusters, xy=(result[member, 0], result[member, 1]))
    plt.show()


if __name__ == '__main__':
    data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""
    a = data.split(',')
    dataset = [(float(a[i]), float(a[i + 1])) for i in range(1, len(a) - 1, 3)]
    dataset = np.array(dataset)
    # dataset = generate_random_number()
    cluster_labels,data = DBSCAN_algorithm(dataset,0.11,5)
    # print(cluster_labels,np.array(data_dict.values()))
    inc = []
    for x,y in cluster_labels.items():
        inc.extend(y)
    print('inc', len(inc))
    draw_graphy(cluster_labels,data)

