KNN
=======
利用KNN实现判断海伦对男生的判断标准和手写体数字类别判断
-------
# k-近邻算法实战之约会网站配对效果判定
## 1、实战背景
>  海伦女士一直使用在线约会网站寻找适合自己的约会对象。尽管约会网站会推荐不同的任选，但她并不是喜欢每一个人。经过一番总结，她发现自己交往过的人可以进行如下分类：
 *不喜欢的人
 *魅力一般的人
 *极具魅力的人
海伦收集约会数据已经有了一段时间，她把这些数据存放在文本文件datingTestSet.txt中，每个样本数据占据一行，总共有1000行。
海伦收集的样本数据主要包含以下3种特征：
 *每年获得的飞行常客里程数
 *玩视频游戏所消耗时间百分比
 *每周消费的冰淇淋公升数
## 2、准备数据：数据解析
在将上述特征数据输入到分类器前，必须将待处理的数据的格式改变为分类器可以接收的格式。分类器接收的数据是什么格式的？从上小结已经知道，要将数据分类两部分，即特征矩阵和对应的分类标签向量。在kNN_test02.py文件中创建名为file2matrix的函数，以此来处理输入格式问题。 将datingTestSet.txt放到与kNN_test02.py相同目录下，编写代码如下：
```
# -*- coding: UTF-8 -*-
import numpy as np
"""
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力
 
Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量
 
Modify:
    2017-03-24
"""
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    arrayOLines = fr.readlines()
    #得到文件行数
    numberOfLines = len(arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines,3))
    #返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector
 
"""
函数说明:main函数
 
Parameters:
    无
Returns:
    无
 
Modify:
    2017-03-24
"""
if __name__ == '__main__':
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    print(datingDataMat)
    print(datingLabels)
```
## 3、分析数据：数据可视化
## 4、准备数据：数据归一化
     表2.1给出了四组样本，如果想要计算样本3和样本4之间的距离，可以使用欧式距离公式计算。
>![表2.1 约会网站样本数据](https://cuijiahua.com/wp-content/uploads/2017/11/ml_1_10.jpg)
## 5、测试算法：验证分类器
## 6、使用算法：构建完整可用系统
# k-近邻算法实战之sklearn手写数字识别
