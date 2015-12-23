#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print(len(enron_data))
count = 0
jp_stock = 0
quantify_salary = 0
known_email = 0
total_payments = 0
for k in enron_data:
    if "lay" in k.lower() or "skilling" in k.lower() or "fastow" in k.lower():
        print("Total payments %s = %s" % (k, enron_data[k]["total_payments"]))
    if "prentice james" in k.lower():
        print("James Prentice stock value %s" % enron_data[k]['total_stock_value'])
    if "colwell wesley" in k.lower():
        print("Wesley Colwell mails nr %s to POI" % enron_data[k]['from_this_person_to_poi'])
    if "skilling jeffrey" in k.lower():
        jef_obj = enron_data[k]
        for z in jef_obj:
            print("jef_obj[%s] = %s" % (z, jef_obj[z]))
    if enron_data[k]['poi']:
        count = count + 1
    if not enron_data[k]['salary'] == 'NaN':
        quantify_salary += 1
    if not enron_data[k]['email_address'] == 'NaN':
        known_email += 1
    if enron_data[k]['total_payments'] == 'NaN' and enron_data[k]['poi']:
        total_payments += 1

percentage = (float(total_payments)/float(len(enron_data)))*100.
print("Total payments not known %s" % total_payments)
print("Known salary = %s" % quantify_salary)
print("Known email = %s" % known_email)

print("POI number %s" % count)

import sys
sys.path.append("../tools/")
from feature_format import *

feature_list = ["poi", "salary", "bonus", "total_payments"]
data_array = featureFormat( enron_data, feature_list )
label, features = targetFeatureSplit(data_array)
