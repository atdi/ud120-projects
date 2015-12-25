#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop( "TOTAL", 0 )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
error_array = []
for k in data_dict:
    if not data_dict[k]["salary"] == "NaN" and not data_dict[k]["bonus"] == "NaN":
        error_array.append((k, data_dict[k]["salary"] + data_dict[k]["bonus"]))

error_array.sort(key=lambda tup: tup[1])
print(error_array[len(error_array) - 1])
print(error_array[len(error_array) - 2])
print(error_array[len(error_array) - 3])
print(error_array[len(error_array) - 4])

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()