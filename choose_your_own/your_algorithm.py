#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################
### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


def nearest_neighbors():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf = clf.fit(features_train, labels_train)
    from sklearn.metrics import accuracy_score
    pred = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    print("KNN = %s" % accuracy)
    return clf


def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(criterion="gini", min_samples_split=50, warm_start=False)
    for i in range(0,1000):
        clf = clf.fit(features_train, labels_train)
        from sklearn.metrics import accuracy_score
        pred = clf.predict(features_test)
        accuracy = accuracy_score(labels_test, pred)
        print("Random forest = %s" % accuracy)
    return clf


def ada_boost():
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(learning_rate=0.5, n_estimators=50)
    clf = clf.fit(features_train, labels_train)
    from sklearn.metrics import accuracy_score
    pred = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    print("AdaBoost = %s" % accuracy)
    return clf


def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, pred)
    print("Naive bayes = %s " % accuracy)


def svm():
    from sklearn.svm import SVC
    clf = SVC(C=10000, kernel="poly", degree=1)
    #features_train = features_train[:len(features_train)/100]
    #labels_train = labels_train[:len(labels_train)/100]
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, pred)
    print("SVM = %s" % accuracy)


def dt():
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(min_samples_split=50)
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, pred)
    print("DT = %s" % accuracy)

#nearest_neighbors()
#ada_boost()
random_forest()
#naive_bayes()
#svm()
#dt()

def display_picture(clf):
    try:
       prettyPicture(clf, features_test, labels_test)
    except NameError:
       pass
