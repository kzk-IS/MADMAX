import numpy as np
import pandas as pd
#import h5py
#import tensorflow as tf
#from sklearn.inspection import permutation_importance
#from sklearn.model_selection import KFold
#from art.utils import to_categorical
import json
import math
import sys
import argparse
import os
from elm_model import ExtremeLearningMachine
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score

import random
import csv
import datetime

def make_label(label):
    if label:
        return [0,1]
    else:
        return [1,0]

def standard_trans(x_train,x_test):
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    x_train_std = stdsc.fit_transform(x_train)
    x_test_std = stdsc.fit_transform(x_test)
    return x_train_std.astype(np.float64),x_test_std.astype(np.float64),(stdsc.mean_,stdsc.var_)


n_unit = 600
start_time = datetime.datetime(2020,8,25,15,15,00)

end_time = datetime.datetime(2020,10,28,12,1,13)
#end_time = datetime.datetime(2020,8,25,16,20,15)

df = pd.read_csv("./allow25000.csv")
allow_list = df[["length", "n_ns", "n_constant_chars", "n_vowel_chars", "life_time", "num_ratio", "n_labels", "mean_TTL", "n_constants", "n_mx"]].values
np.random.shuffle(allow_list)

df = pd.read_csv("./deny_with_time.csv")
deny_time = df["time"].values
deny_list = df[["length", "n_ns", "n_constant_chars", "n_vowel_chars", "life_time", "num_ratio", "n_labels", "mean_TTL", "n_constants", "n_mx"]].values

x_train = np.concatenate([allow_list[:20000], deny_list[:20000]])
x_test = np.concatenate([allow_list[20000:25000], deny_list[20000:25000]])

y_train = np.concatenate([np.zeros(20000), np.ones(20000)]) 
y_test = np.concatenate([np.zeros(5000), np.ones(5000)])

x_train_std, x_test_std, _ = standard_trans(x_train,x_test)
y_train = np.array([make_label(l) for l in y_train])
y_test = np.array([make_label(l) for l in y_test])
y_test_true = np.argmax(y_test,axis=1)

acc_list = []
pre_list = []
rec_list = []
f1_list = []

model0 = ExtremeLearningMachine(n_unit=n_unit, activation=None)
model0.fit(x_train_std, y_train)
y_test_pred = np.argmax(model0.transform(x_test_std), axis=1)
acc_list.append(accuracy_score(y_test_true,y_test_pred))
pre_list.append(precision_score(y_test_true,y_test_pred))
rec_list.append(recall_score(y_test_true,y_test_pred))
f1_list.append(f1_score(y_test_true,y_test_pred))

model3 = ExtremeLearningMachine(n_unit=n_unit, activation=None)
model3.fit(x_train_std, y_train)
y_test_pred = np.argmax(model3.transform(x_test_std), axis=1)
acc_list.append(accuracy_score(y_test_true,y_test_pred))
pre_list.append(precision_score(y_test_true,y_test_pred))
rec_list.append(recall_score(y_test_true,y_test_pred))
f1_list.append(f1_score(y_test_true,y_test_pred))

model6 = ExtremeLearningMachine(n_unit=n_unit, activation=None)
model6.fit(x_train_std, y_train)
y_test_pred = np.argmax(model6.transform(x_test_std), axis=1)
acc_list.append(accuracy_score(y_test_true,y_test_pred))
pre_list.append(precision_score(y_test_true,y_test_pred))
rec_list.append(recall_score(y_test_true,y_test_pred))
f1_list.append(f1_score(y_test_true,y_test_pred))

model9 = ExtremeLearningMachine(n_unit=n_unit, activation=None)
model9.fit(x_train_std, y_train)
y_test_pred = np.argmax(model9.transform(x_test_std), axis=1)
acc_list.append(accuracy_score(y_test_true,y_test_pred))
pre_list.append(precision_score(y_test_true,y_test_pred))
rec_list.append(recall_score(y_test_true,y_test_pred))
f1_list.append(f1_score(y_test_true,y_test_pred))

with open('./results10/results10acc.csv', 'a') as fw:
	writer = csv.writer(fw)
	writer.writerow(acc_list)

with open('./results10/results10pre.csv', 'a') as fw:
	writer = csv.writer(fw)
	writer.writerow(pre_list)

with open('./results10/results10rec.csv', 'a') as fw:
	writer = csv.writer(fw)
	writer.writerow(rec_list)

with open('./results10/results10f1.csv', 'a') as fw:
	writer = csv.writer(fw)
	writer.writerow(f1_list)

current_time = start_time

index = 20000

previous_i = 20000
l_count = 0
count6 = 0
count9 = 0

delta_time = datetime.timedelta(seconds=30)

while current_time < end_time:
	
	i = previous_i
	while True:
		if datetime.datetime.strptime(deny_time[i], "%Y-%m-%dT%H:%M:%SZ") > current_time:
			break
		else:
			i += 1

	if i == previous_i:
		pass
	else:
		print("---------------")
		print(current_time)
		print("i: "+str(i))
		print("30s Larning")
		l_count += 1

		x_train = np.concatenate([allow_list[:20000], deny_list[i-20000:i]])
		x_test = np.concatenate([allow_list[20000:25000], deny_list[i:i+5000]])

		x_train_std,x_test_std, _ = standard_trans(x_train,x_test)

		model3.fit(x_train_std,y_train)	

		if l_count%20160 == 0:
			print("60s Larning")
			count6 += 1
			model6.fit(x_train_std,y_train)

		if l_count%40320 == 0:
			print("90s Larning")
			count9 += 1
			model9.fit(x_train_std,y_train)

		acc_list = []
		pre_list = []
		rec_list = []
		f1_list = []

		y_test_pred = np.argmax(model0.transform(x_test_std), axis=1)
		acc_list.append(accuracy_score(y_test_true,y_test_pred))
		pre_list.append(precision_score(y_test_true,y_test_pred))
		rec_list.append(recall_score(y_test_true,y_test_pred))
		f1_list.append(f1_score(y_test_true,y_test_pred))

		y_test_pred = np.argmax(model3.transform(x_test_std), axis=1)
		acc_list.append(accuracy_score(y_test_true,y_test_pred))
		pre_list.append(precision_score(y_test_true,y_test_pred))
		rec_list.append(recall_score(y_test_true,y_test_pred))
		f1_list.append(f1_score(y_test_true,y_test_pred))

		y_test_pred = np.argmax(model6.transform(x_test_std), axis=1)
		acc_list.append(accuracy_score(y_test_true,y_test_pred))
		pre_list.append(precision_score(y_test_true,y_test_pred))
		rec_list.append(recall_score(y_test_true,y_test_pred))
		f1_list.append(f1_score(y_test_true,y_test_pred))

		y_test_pred = np.argmax(model9.transform(x_test_std), axis=1)
		acc_list.append(accuracy_score(y_test_true,y_test_pred))
		pre_list.append(precision_score(y_test_true,y_test_pred))
		rec_list.append(recall_score(y_test_true,y_test_pred))
		f1_list.append(f1_score(y_test_true,y_test_pred))

		with open('./results10/results10acc.csv', 'a') as fw:
			writer = csv.writer(fw)
			writer.writerow(acc_list)

		with open('./results10/results10pre.csv', 'a') as fw:
			writer = csv.writer(fw)
			writer.writerow(pre_list)

		with open('./results10/results10rec.csv', 'a') as fw:
			writer = csv.writer(fw)
			writer.writerow(rec_list)

		with open('./results10/results10f1.csv', 'a') as fw:
			writer = csv.writer(fw)
			writer.writerow(f1_list)

		previous_i = i
	current_time += delta_time

print("learning_count: " + str(l_count))
print("learning_count: " + str(count6))
print("learning_count: " + str(count9))
