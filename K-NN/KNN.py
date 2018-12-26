# -*- coding:UTF-8 -*-
from numpy import *
import numpy as np
import operator

def createDataset():
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    labels=['爱情','爱情','动作','动作']
    return group,labels
'''
if __name__=='__main__':
    group,labels=createDataset()
    print (group)
    print (labels)
'''


def kNN_classify(newInput,dataSet,labels,k):
    numSamples=dataSet.shape[0] #shape[0]表示行数

    # step 1: 计算距离
    diffMat=tile(newInput,(numSamples,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1) # 按行累加
    distances=sqDistances**0.5

    # step 2: 对距离排序
    sortedDistIndices=distances.argsort() #返回distances中元素从小到大排序后的索引值
    classCount={}
    for i in range(k):
        # step 3: 选择k个最近邻
        votelabel=labels[sortedDistIndices[i]]
        # step 4: 计算k个最近邻中各类别出现的次数
        classCount[votelabel]=classCount.get(votelabel,0)+1

    # step 5: 返回出现次数最多的类别标签
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__=='__main__':
    #创建数据集
    group,labels=createDataset()
    #测试集
    test=[101,20]
    #测试分类
    test_class=kNN_classify(test,group,labels,3)
    print(test_class)
