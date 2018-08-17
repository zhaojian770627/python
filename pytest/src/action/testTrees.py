import matplotlib
from numpy import*

import trees
import matplotlib.pyplot as plt

myDat, labels = trees.createDataSet()
print(trees.splitDataSet(myDat, 0, 1))
