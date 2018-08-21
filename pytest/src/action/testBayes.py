import bayes

listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print(bayes.setOfWords2Vec(myVocabList, listOPosts[0]))
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
    
p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
print(pAb)