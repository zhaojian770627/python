# 朴素贝叶斯
# https://blog.csdn.net/mlljava1111/article/details/50512913
from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字, 0代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    # 创建一个空集
    vocabSet = set([])
    for document in dataSet:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 词表到向量的转换函数
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为０的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec


# trainCategory 是否侮辱性文档
# 输入trainMatrix：词向量数据集
# 输入trainCategory：数据集对应的类别标签
# 输出p0Vect：词汇表中各个单词在正常言论中的类条件概率密度
# 输出p1Vect：词汇表中各个单词在侮辱性言论中的类条件概率密度
# 输出pAbusive：侮辱性言论在整个数据集中的比例
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 计算p(ci) 属于侮辱性文档的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化概率
    # 将所有词的出现数初始化为1,防止出现概率值为0的情况
    p0Num = ones(numWords);
    p1Num = ones(numWords)
    # 并将分母初始化为2
    p0Denom = 2.0
    p1Denom = 2.0
    # 要遍历训练集trainMatrix中的所有文档。一旦某个词语(侮辱性或正常词语）在某一文档中出现，
    # 则该词对应的个数(p1Num或者p1Num)就加1，
    # 而且在所有的文档中，该文档的总词数也相应加1.
    for i in range(numTrainDocs):
        # 向量相加
        # 如果是侮辱性文档,计算
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # p(w|ci)
            p1Denom += sum(trainMatrix[i])  # p(w)
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 防止很小的数相乘，造成为0的情况
    # 将该词条的数目除以总词条数目得到条件概率
    # 计算　p(ci|w) -- ? p1Num->p(w|c)  p1Denom -> p(w)
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 对应元素相乘，然后相加
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
