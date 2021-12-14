import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from sklearn import datasets

file_CSV = open("kNNData.csv")
data_CSV = csv.reader(file_CSV)
list_CSV = list(data_CSV)
data=[]
for i in list_CSV:
    # print(i)
    data.append(i)

# del data[0]
# print(data)

test=['897471', '4', '8', '8', '5', '4', '5', '10', '4', '1', '4']
test2=['897471', '4', '8', '8', '4', '4', '5', '80', '4', '1', '4']

def euclid_metrics(a,b):
    len_vector=len(a)
    s=0
    for i in range(len_vector):
        if((i>0 )and (i<10)):
            s=s+math.pow(int(a[i])-int(b[i]),2)
    
    return math.sqrt(s)

# print(euclid_metrics(test,test2))

def kNN_prediction(dataset,test,k=35):
    metrics=[]
    for i in range(len(dataset)):
        # print([dataset[i][0],euclid_metrics(dataset[i],test)])
        metrics.append([dataset[i][10],euclid_metrics(dataset[i],test)])
        
    sorted_list = sorted(metrics, key=lambda x:x[1])
    # print(sorted_list)
    list_result=[]
    for j in range(k):
        list_result.append(sorted_list[j])
    two=0
    four=0
    for j in list_result:
        if(j[0]=='4'):
            four=four+1
        else:
            two=two+1
    two=two/k
    four=four/k
    pr_result=int(test[10])
    result=''
    result=two
    prediction=2
    if(pr_result==2):
        return two
    if(pr_result==4):
        return four
    if(four>two):
        prediction=4
        result=four
    return prediction,result
# print(kNN_prediction(data,test))


def find_best_k():
    for i in range(30,80,5):
        if(i!=0):
            print(i, kNN_prediction(data,test,i))
    return 0
find_best_k()
# iris = datasets.load_iris()

# df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# df['target'] = iris.target
# # print(df.head())

# # Separate X and y data
# X = df.drop('target', axis=1)
# y = df.target

# # Calculate distance between two points

# def minkowski_distance(a, b, p=2):
    
#     # Store the number of dimensions
#     dim = len(a)
    
#     # Set initial distance to 0
#     distance = 0
    
#     # Calculate minkowski distance using parameter p
#     for d in range(dim):
#         distance += abs(a[d] - b[d])**p
        
#     distance = distance**(1/p)
    
#     return distance


# # Test the function

# minkowski_distance(a=X.iloc[0], b=X.iloc[1], p=2)

# # Define an arbitrary test point

# test_pt = [4.8, 2.7, 2.5, 0.7]

# # Calculate distance between test_pt and all points in X

# distances = []

# for i in X.index:
    
#     distances.append(minkowski_distance(test_pt, X.iloc[i]))
    
# df_dists = pd.DataFrame(data=distances, index=X.index, columns=['dist'])
# df_dists.head()


# # Find the 5 nearest neighbors
# df_nn = df_dists.sort_values(by=['dist'], axis=0)[:5]
# df_nn
# # print(df_nn)

# from collections import Counter

# # Create counter object to track the labels

# counter = Counter(y[df_nn.index])
# # counter
# # print(counter)

# # Get most common label of all the nearest neighbors

# counter.most_common()[0][0]
# print(counter.most_common()[0][0])

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Split the data - 75% train, 25% test

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
#                                                    random_state=1)

# # Scale the X data

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# def knn_predict(X_train, X_test, y_train, y_test, k, p):
    
#     # Counter to help with label voting
#     from collections import Counter
    
#     # Make predictions on the test data
#     # Need output of 1 prediction per test data point
#     y_hat_test = []

#     for test_point in X_test:
#         distances = []

#         for train_point in X_train:
#             distance = minkowski_distance(test_point, train_point, p=p)
#             distances.append(distance)
        
#         # Store distances in a dataframe
#         df_dists = pd.DataFrame(data=distances, columns=['dist'], 
#                                 index=y_train.index)
        
#         # Sort distances, and only consider the k closest points
#         df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

#         # Create counter object to track the labels of k closest neighbors
#         counter = Counter(y_train[df_nn.index])

#         # Get most common label of all the nearest neighbors
#         prediction = counter.most_common()[0][0]
        
#         # Append prediction to output list
#         y_hat_test.append(prediction)
        
#     return y_hat_test


# # Make predictions on test dataset
# y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=5, p=2)

# # print(y_hat_test)

# # Get test accuracy score

# from sklearn.metrics import accuracy_score

# # print(y_test)
# # print(y_hat_test)
# # print(accuracy_score(y_test, y_hat_test))

# # Obtain accuracy score varying k from 1 to 99

# accuracies = []

# for k in range(1,100):
#     y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k, p=2)
#     accuracies.append(accuracy_score(y_test, y_hat_test))

# # Plot the results 
# print(accuracies)
# fig, ax = plt.subplots(figsize=(8,6))
# ax.plot(range(1,100), accuracies)
# ax.set_xlabel('# of Nearest Neighbors (k)')
# ax.set_ylabel('Accuracy (%)')