# -*- coding:UTF-8 -*-
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from KNN import *

def file2matrix(filename):
    fr=open(filename)
    array0Lines=fr.readlines()
    numberofLines=len(array0Lines)
    returnMat=np.zeros((numberofLines,3))

    #返回的分类标签向量
    classLabelVector=[]
    #行的索引值
    index=0
    for line in array0Lines:
        #去除首尾空格
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        if listFromLine[-1]=='didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1]=='smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1]=='largeDoses':
            classLabelVector.append(3)
        index +=1
    return returnMat,classLabelVector


def showdatas(datingDataMat,datingLabels):
    font=FontProperties(fname=r"/usr/share/fonts/truetype/ancient-scripts/Symbola_hint.ttf",size=14)
    fig,axs=plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))
    numberoflabels=len(datingLabels)
    LabelsColors=[]
    for i in datingLabels:
        if i==1:
            LabelsColors.append('black')
        if i==2:
            LabelsColors.append('orange')
        if i==3:
            LabelsColors.append('red')
    axs[0][0].scatter(x=dataingDataMat[:,0],y=dataingDataMat[:,1],color=LabelsColors,s=15,alpha=0.5)
    axs0_title_text=axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text=axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text=axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    plt.setp(axs0_title_text,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel_text,size=9,weight='bold',color='black')
    plt.setp(axs0_ylabel_text,size=9,weight='bold',color='black')

    axs[0][1].scatter(x=dataingDataMat[:, 0], y=dataingDataMat[:, 2], color=LabelsColors, s=15, alpha=0.5)
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=9, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=9, weight='bold', color='black')

    axs[1][0].scatter(x=dataingDataMat[:, 1], y=dataingDataMat[:, 2], color=LabelsColors, s=15, alpha=0.5)
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=9, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=9, weight='bold', color='black')


    #设置图例
    didntlike=mlines.Line2D([],[],color='black',marker='.',markersize=6,label='didntlike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='+', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='o', markersize=6, label='largeDoses')

    #添加图例
    axs[0][0].legend(handles=[didntlike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntlike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntlike, smallDoses, largeDoses])
    plt.show()


def autoNorm(dataSet):
    minVals=dataSet.min(0)#当参数为0是，min()函数返回每一列的最小值
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataSet))
    m=dataSet.shape[0]#看行数（也就是第一个维度）
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    filename='datingTestSet.txt'
    dataingDataMat, dataingLabels = file2matrix(filename)
    hoRatio=0.10
    normMat, ranges, minVals = autoNorm(dataingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult=kNN_classify(normMat[i,:],normMat[numTestVecs:m,:],dataingLabels[numTestVecs:m],4)
        print("分类结果：%d\t真实类别：%d"%(classifierResult,dataingLabels[i]))
        if classifierResult!=dataingLabels[i]:
            errorCount+=1.0
    print("错误率：%f%%"%(errorCount/float(numTestVecs)*100))  #最后二个连续的%%就是实际输出一个%符号

def classifyPerson():
    resultList=['讨厌','有些喜欢','非常喜欢']
    precentTats=float(input("玩视频游戏所消耗时间占比:"))
    ffMile=float(input("每年获得的飞行常客里程数:"))
    iceCream=float(input("每周消费的冰激淋公升数:"))
    filename="datingTestSet.txt"
    dataingDataMat, dataingLabels = file2matrix(filename)
    normMat,ranges, minVals = autoNorm(dataingDataMat)
    inArr=np.array([ffMile,precentTats,iceCream])
    norminArr=(inArr-minVals)/ranges
    classifierResult=kNN_classify(norminArr,normMat,dataingLabels,3)
    print("你可能%s这个人"%(resultList[classifierResult-1]))

if __name__=='__main__':
    classifyPerson()

'''
if __name__=='__main__':
    datingClassTest()
'''

'''
if __name__=='__main__':
    filename='datingTestSet.txt'
    dataingDataMat,dataingLabels=file2matrix(filename)
    normDataSet,ranges,minVals=autoNorm(dataingDataMat)
    print(normDataSet)
    print(ranges)
    print(minVals)
    #showdatas(dataingDataMat,dataingLabels)
    #print(dataingDataMat)
    #print(dataingLabels)
'''

