import csv
import numpy as np

test_path = "../elmo/vec_test_y.csv"
train_path = "../elmo/vec_train_y.csv"

f_test = open(test_path, 'r')
r_test = csv.reader(f_test)

f_train = open(train_path, 'r')
r_train = csv.reader(f_train)

test_arr = []

for line in r_test:
        result = line[0]
        test_arr.append(result)

train_arr = []

for line in r_train:
        result = line[0]
        train_arr.append(result)

result_arr1 = np.array(test_arr)
result_arr2 = np.array(train_arr)

np.save("../elmo/y_test.npy", result_arr1)
np.save("../elmo/y_train.npy", result_arr2)
