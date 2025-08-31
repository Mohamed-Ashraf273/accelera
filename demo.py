from aistudio.src.core.graph import Graph

from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[0,0],[1,1],[2,2],[3,3]])
y = np.array([0,0,1,1])

clf = LogisticRegression().fit(X, y)
g = Graph()
g.add("logreg", LogisticRegression().fit, X, y).add("predict", clf.predict, X)
print("Result:", g.compute())