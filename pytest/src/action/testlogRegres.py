import logRegres

dataArr, labelMat = logRegres.loadDataSet()
# print(logRegres.gradAscent(dataArr, labelMat))
weights = logRegres.gradAscent(dataArr, labelMat)
logRegres.plotBestFit(weights.getA())
