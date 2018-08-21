import trees
import treePlotter

fr = open("lenses.txt")
lenses = [ inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
print(lensesTree)
result = trees.classify(lensesTree, ['age', 'prescript', 'astigmatic', 'tearRate'], ['young', 'myope', 'no', 'normal'])
print(result)
# treePlotter.createPlot(lensesTree)

