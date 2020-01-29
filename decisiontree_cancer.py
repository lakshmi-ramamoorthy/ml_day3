# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:16:54 2020

@author: RPS
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:01:51 2017

@author: BALASUBRAMANIAM
"""

# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import pandas as pd

dataset=pd.read_csv("breast-cancer-wisconsin.csv")
X = dataset.iloc[:, 1:11].values
y = dataset.iloc[:,10:11].values

print(type(X))
print(type(y))

features=["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses"]
target=["Type2","Type4"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit a CART model to the data

model = DecisionTreeClassifier(criterion='entropy',
                       max_depth=5, random_state=0)
model.fit(X_train, y_train)
#print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
print(expected)
print(predicted)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# Create DOT data
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=features,  
                                class_names=target)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("cancer.pdf")
