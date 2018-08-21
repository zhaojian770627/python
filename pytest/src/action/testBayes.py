import bayes

listOPosts, listClasses = bayes.loadDataSet()
print(listOPosts)
print(listClasses)
myVocabList = bayes.createVocabList(listOPosts)
print(myVocabList)
print(bayes.setOfWords2Vec(myVocabList, listOPosts[0]))
