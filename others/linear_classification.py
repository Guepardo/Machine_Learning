import sklearn           as sklearn
import numpy             as np

from sklearn                  import datasets
from sklearn.cross_validation import train_test_split
from sklearn                  import preprocessing

iris = datasets.load_iris()

X_iris, y_iris = iris.data, iris.target

# print(X_iris.shape, y_iris.shape)
# print(X_iris[0], y_iris[0])

#Recupera o dataset com apenas os dois primeiros atributos
X, y = X_iris[:, :2], y_iris

#Dividir o datase em consunto de treinamento e de teste
#O conjunto de teste será escolido aleatoriamente em 25%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# print(X_train.shape, y_train.shape)

#Normalizando features
scaler  = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

import matplotlib.pyplot as plt

colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)): 
	xs = X_train[:, 0][y_train == i]
	ys = X_train[:, 1][y_train == i] 
	plt.scatter(xs, ys, c=colors[i])

plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

from sklearn import linear_model

clf = linear_model.SGDClassifier()
clf.fit(X_train, y_train)

# print( clf.coef_ )
# print( clf.intercept_ )

# print( clf.predict(scaler.transform([[4.7, 3.1]])) )

print('Decision function')
print( clf.decision_function(scaler.transform([[4.7, 3.1]])))

from sklearn import metrics

#Análise de acerto com o conjunto de treinamento
print("Análise de acerto com o conjunto de treinamento")
y_train_pred = clf.predict(X_train)
print (metrics.accuracy_score(y_train, y_train_pred))

#Análise de acerto com o conjunto de teste
print("Análise de acerto com o conjunto de teste")
y_pred = clf.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred))