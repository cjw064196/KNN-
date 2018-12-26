import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN


def img2vector(filename):
    # 创建1x1024零向量
    returnVec=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVec[0,32*i+j]=int(lineStr[j])
    return returnVec

def handwritingClassTest():
    hwLabels=[]
    traingFileList=listdir('trainingDigits')
    m=len(traingFileList)
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=traingFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:]=img2vector('trainingDigits/%s'%(fileNameStr))
    # 构建kNN分类器
    neigh=KNN(n_neighbors=3,algorithm='auto')#n_neighbors就是k-NN的k的值，选取最近的k个点,algorithm：快速k近邻搜索算法，默认参数为auto，可以理解为算法自己决定合适的搜索算法
    # 拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat,hwLabels)
    errorCount=0.0
    testFileList=listdir('testDigits')
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest=img2vector('testDigits/%s'%(fileNameStr))
        classifierResult=neigh.predict(vectorUnderTest)
        print('分类返回结果为%d\t真实结果为%d'%(classifierResult,classNumber))
        if (classifierResult!=classNumber):
            errorCount+=1.0
    print('总共错了%d个数据\n错误率为%f%%'%(errorCount,errorCount/mTest*100))

if __name__=='__main__':
    handwritingClassTest()