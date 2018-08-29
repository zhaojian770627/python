from numpy import *
import logRegres

dataArr, labelMat = logRegres.loadDataSet()
# print(logRegres.gradAscent(dataArr, labelMat))
# weights = logRegres.gradAscent(dataArr, labelMat)
weights = logRegres.stocGradAscent0(array(dataArr), labelMat)
logRegres.plotBestFit(weights)
