import trees
import treePlotter

myDat, labels = trees.createDataSet()
myTree = treePlotter.retrieveTree(0)
trees.storeTree(myTree, 'tree.txt')
myTree2 = trees.grabTree('tree.txt')
print(myTree2)

