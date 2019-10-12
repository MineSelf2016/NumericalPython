#%%[markdown]
# # 使用 K 近邻法改进约会网站配对效果

#%%
import numpy as np
import pandas as pd 

raw_data = pd.read_csv("Chapter2/datasets/datingTestSet.txt", sep="\t", header=None)
raw_data

#%%
x = np.array(raw_data.iloc[:, :3])
y = np.array(raw_data.iloc[:, -1])

#%%[markdown]
# Normalize

#%%
for i in range(len(x[0])):
    x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min())

def func(str):
    if str == "didntLike":
        return 1
    elif str == "smallDoses":
        return 2
    else:
        return 3


y = np.array(list(map(func, y)))

#%%
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x[:, 0], x[:, 1], 15.0 * y, 15.0 * y)
plt.show()

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 

knnclf = KNeighborsClassifier(n_neighbors=3)
x_train, x_test, y_train, y_test = train_test_split(x, y)
knnclf.fit(x_train, y_train)
y_pred = knnclf.predict(x_test)
total_number = len(y_test)
error_number = np.sum((y_pred != y_test))
error_rate = error_number / total_number
"total point number: {} , error point number: {}, error rate : {}".format(total_number, error_number, error_rate)

#%%
