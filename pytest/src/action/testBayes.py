import bayes

listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print(myVocabList)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
    
p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
print('-------------------p0V----------------------')
print(p0V)
print('-------------------p1V----------------------')
print(p1V)
print('-------------------pAb----------------------')
print(pAb)