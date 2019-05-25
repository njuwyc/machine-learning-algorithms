import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def minmax_scale(X):
    ma=X.max(axis=0)
    mi=X.min(axis=0)
    return (X-mi)/(ma-mi)
    

def get_distance(instance,example):
    return sum((instance-example)**2)
    

def choose_k_neighbours(instance,k,X_train):
    distance=np.array([get_distance(instance,example) for example in X_train])
    k_min_index=distance.argsort()[:k]
    return k_min_index      
    

def knn(X_train,y_train,X_valid,y_valid,k):
    result=[]
    for instance in X_valid:
        k_min_index=choose_k_neighbours(instance,k,X_train)
        y_neighbours=list(y_train[k_min_index])
        label=max(set(y_neighbours),key=y_neighbours.count) 
        result.append(label)
    return np.array(result)
        
        
iris = datasets.load_iris()
X=iris.data
y=iris.target
X_train_all,X_test,y_train_all,y_test=train_test_split(X,y,test_size=0.3) 
X_train_all=minmax_scale(X_train_all)   
X_test=minmax_scale(X_test)
f=5   
best_accuracy=0
for k in range(1,11):
    accuracy=0
    skf = StratifiedKFold(n_splits=f)
    for train_index,valid_index in skf.split(X_train_all,y_train_all):
        #print(train_index,valid_index)
        X_train=X_train_all[train_index]
        y_train=y_train_all[train_index]
        X_valid=X_train_all[valid_index]
        y_valid=y_train_all[valid_index]
        y_valid_pred=knn(X_train,y_train,X_valid,y_valid,k)
        accuracy+=np.mean(y_valid_pred==y_valid)
    accuracy=accuracy/f
    print(accuracy)
    if accuracy>best_accuracy:
        best_accuracy=accuracy
        best_k=k

print(best_k)
y_test_pred=knn(X_train_all,y_train_all,X_test,y_test,best_k)
test_score=np.mean(y_test_pred==y_test)
print('测试集精度：',test_score)


    