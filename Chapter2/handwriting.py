
#%%
import os
import numpy as np 
import cv2
from sklearn.neighbors import KNeighborsClassifier

#%%
def list2data(base_file_path, filename_list):
    x_train = np.zeros((len(filename_list), 32*32))
    y_train = np.array([0 for i in range(len(filename_list))])
    for i, filename in enumerate(filename_list):
        with open(base_file_path + filename) as f:
            y_train[i] = (filename.split("_")[0])
            lines = f.readlines()
            int_line = []
            for line in lines:
                for word in range(32):
                    int_line.append(int(line[word]))
            x_train[i] = int_line
    return x_train, y_train


#%%
base_path = "Chapter2/datasets/" 
x_file_path = base_path + "trainingDigits/"
y_file_path = base_path + "testDigits/"
x_filename_list = os.listdir(x_file_path)
y_filename_list = os.listdir(y_file_path)

x_filename_list[:5], y_filename_list[:5]


#%%
x_train, y_train = list2data(x_file_path, x_filename_list)
x_test, y_test = list2data(y_file_path, y_filename_list)


#%%
knnclf = KNeighborsClassifier(3)
knnclf.fit(x_train, y_train)


#%%
y_pred = knnclf.predict(x_test)

#%%
def show_image(xx):
    for item in xx:
        for i, w in enumerate(item):
            if w > 0:
                item[i] = 255
        item = item.reshape((32, 32))
        cv2.imshow("digit_" + str(i), item)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


#%%
total_number = len(y_test)
result_label = (y_pred != y_test)


error_number = np.sum(result_label)
error_rate = error_number / total_number
"total point number: {} , error point number: {}, error rate : {}".format(total_number, error_number, error_rate)


#%%
arg_result = []
for i, element in enumerate(result_label):
    # print(element)
    if element:
        print("the classifier came back with: ", y_pred[i], "the real class is: ", y_test[i])
        arg_result.append(x_test[i])

#%%
show_image(arg_result)

#%%
