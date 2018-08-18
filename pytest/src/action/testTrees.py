import matplotlib
from numpy import*

import trees
import matplotlib.pyplot as plt

myDat, labels = trees.createDataSet()
print(myDat)
print(trees.splitDataSet(myDat, 0, 1))
