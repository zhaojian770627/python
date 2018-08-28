import bayes
import feedparser

# listOPosts, listClasses = bayes.loadDataSet()
# myVocabList = bayes.createVocabList(listOPosts)
# print(myVocabList)
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
#     
# p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
# print('-------------------p0V----------------------')
# print(p0V)
# print('-------------------p1V----------------------')
# print(p1V)
# print('-------------------pAb----------------------')
# print(pAb)
# bayes.testingNB()
# emailText = open('/home/zj/sourcecode/machinelearninginaction/Ch04/email/ham/6.txt', encoding="latin-1").read()
# print(emailText)
# bayes.spamTest()
ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf = feedparser.parse('https://www.state.gov/rss/channels/alldos.xml')
# vocabList, pSF, pNY = bayes.localWords(ny, sf)
# print(len(sf['entries']))
bayes.getTopWords(ny, sf)