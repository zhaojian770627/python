import bayes

listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print(myVocabList)
print(listOPosts[0])
print(bayes.setOfWords2Vec(myVocabList, listOPosts[0]))
