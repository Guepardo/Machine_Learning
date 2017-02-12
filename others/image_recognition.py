import sklearn           as sk
import numpy             as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()

print(faces.DESCR)

print(faces.keys())
print(faces.data.shape)
print(faces.target.shape)

print(np.max(faces.data))
print(np.min(faces.data))
print(np.mean(faces.data))

def print_faces(images, target, top_n): 
	#set up the figure size in inches
	fig = plt.figure(figsize=(12, 12))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
	for i in range(top_n): 
		#plot the images in a matrix of 20x20
		p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
		p.imshow(images[i], cmap=plt.cm.bone)

		#label the imagen with the target value
		p.text(0, 14, str(target[i]))
		p.text(0, 60, str(i))
	# fig.show()
	# input()

print_faces(faces.images, faces.target, 20)

from sklearn.svm import SVC

svc_1 = SVC(kernel='linear')

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0)

from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats 			  import sem

def evaluate_cross_validation(clf, X, y, K):
	#create a k-fold croos validation iterator
	cv = KFold(len(y), K, shuffle=True, random_state=0)
	#by default the score used is the one returned by score method of the estimator (accuracy)
	scores = cross_val_score(clf, X, y, cv=cv)

	print(scores)
	print(("Mean scores: {0:.3f} (+/-{1:3f})").format(np.mean(scores), sem(scores)))

evaluate_cross_validation(svc_1, X_train, y_train, 5)

from sklearn import metrics

def train_and_evaluate(clf, X_train, X_test, y_train, y_test): 
	
	clf.fit(X_train, y_train)

	print("Accuracy on traning set:")
	print(clf.score(X_train, y_train))

	print("Accuracy on test set: ")
	print(clf.score(X_test, y_test))

	y_pred = clf.predict(X_train)

	print("Classification Report: ")
	print(metrics.classification_report(y_test, y_pred))
	print("Confusion Matrix: ")
	print(matrics.confusion_matrix(y_test, y_pred))

train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)