import math
import numpy as np
import pandas as pd

class treenode_continous:
    def __init__(self,feature=None,value=None,label=None):
        self.lchild=None
        self.rchild=None
        self.feature=feature
        self.value=value
        self.label=label
        
    def display(self,ind=0,msg=''):
        if self.lchild==None and self.rchild==None:
            print('   '*ind,msg,str(self.label)+'类')
        else:    
            print('   '*ind,msg,'第'+str(self.feature)+'个属性'+'<='+str(self.value))
            self.lchild.display(ind+1,msg='是')
            self.rchild.display(ind+1,msg='否')
        
    def predict(self,X_test):
        y_test_pred=[]
        for instance in X_test:
            y_test_pred.append(self.predict_one(instance))
        return np.array(y_test_pred)
            
    def predict_one(self,instance):
        if self.lchild==None and self.rchild==None:
            result=self.label
        else:
            if instance[self.feature]<=self.value:
                result=self.lchild.predict_one(instance)
            else:
                result=self.rchild.predict_one(instance)
        return result
    
            
def calculate_entropy(X,y):
    y=list(y)
    classunique=list(set(y))
    entropy=0
    for cla in classunique:
        ratio=(y.count(cla))/len(y)
        entropy=entropy-ratio*math.log(ratio,2)
    return entropy


def split_data_continous(X,y,i,point):
    smaller_X=X[np.where(X[:,i]<point)]
    smaller_y=y[np.where(X[:,i]<point)]
    bigger_X=X[np.where(X[:,i]>point)]
    bigger_y=y[np.where(X[:,i]>point)]
    return smaller_X,smaller_y,bigger_X,bigger_y
            

def get_mean_points(feature_value_set_one):
    mean_points=[]
    feature_value_set_one.sort()
    for i in range(len(feature_value_set_one)-1):
        mean_points.append((feature_value_set_one[i]+feature_value_set_one[i+1])/2)
    return mean_points


def choose_best_split_continous(X,y,feature_value_set):
    feanum=np.shape(feature_value_set)[0]
    old_entropy=calculate_entropy(X,y)
    #print(old_entropy)
    g=0
    for i in range(feanum):
        mean_points=get_mean_points(feature_value_set[i])
        for point in mean_points:
            smaller_X,smaller_y,bigger_X,bigger_y=split_data_continous(X,y,i,point)
            smaller_entropy=calculate_entropy(smaller_X,smaller_y)
            bigger_entropy=calculate_entropy(bigger_X,bigger_y)
            new_entropy=smaller_entropy*len(smaller_X)/len(X)+bigger_entropy*len(bigger_X)/len(X)
            information_gain=old_entropy-new_entropy
            #print('第',i,'个属性，','划分点为',point,'，划分后熵为',new_entropy,'，信息增益为',information_gain,sep='')
            if information_gain>g:
                g=information_gain
                best_feature=i
                best_split_point=point
    return best_feature,best_split_point
                

def get_majority_label(y):
    return pd.Series(y).value_counts().index[0]


def create_tree_continous(X,y):
    if len(set(list(y)))==1:
        leaf=treenode_continous(label=y[0])
        return leaf
    elif len(np.array(pd.DataFrame(X).drop_duplicates()))==1:
        majority_label=get_majority_label(y)
        leaf=treenode_continous(label=majority_label)
        return leaf
    else:
        feanum=np.shape(X)[1]
        feature_value_set=[list(set(X[:,i])) for i in range(feanum)]
        best_feature,best_split_point=choose_best_split_continous(X,y,feature_value_set)
        smaller_X,smaller_y,bigger_X,bigger_y=split_data_continous(X,y,best_feature,best_split_point)
        #print(smaller_X,smaller_y,bigger_X,bigger_y,sep='\n')
        tnode=treenode_continous(best_feature,best_split_point)
        
        if len(smaller_y)==0:
            majority_label=get_majority_label(y)
            leaf=treenode_continous(label=majority_label)
            tnode.lchild=leaf
        else:
            tnode.lchild=create_tree_continous(smaller_X,smaller_y)
        if len(bigger_y)==0:
            majority_label=get_majority_label(y)
            leaf=treenode_continous(label=majority_label)
            tnode.rchild=leaf
        else:
            tnode.rchild=create_tree_continous(bigger_X,bigger_y)
            
        return tnode



'''
X=np.array([[24,40],[53,52],[23,25],[25,77],[32,48],
           [52,110],[22,38],[43,44],[52,27],[48,65]])
y=np.array([1,0,0,1,1,1,1,0,0,1])
rootnode=create_tree_continous(X,y)
'''

'''
X_train=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],
                  [0.556,0.215],[0.403,0.237],[0.481,0.149],[0.437,0.211],
                  [0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],
                  [0.639,0.161],[0.657,0.198],[0.360,0.370],[0.593,0.042],
                  [0.719,0.103]])
y_train=np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
tree=create_tree_continous(X_train,y_train)
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
import time
iris=datasets.load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
start_time=time.time()
mytree=create_tree_continous(X_train,y_train)
y_test_pred=mytree.predict(X_test)
print(time.time()-start_time)
print(np.mean(y_test==y_test_pred))

            

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier().fit(X_train,y_train)
print(model.score(X_test,y_test))
