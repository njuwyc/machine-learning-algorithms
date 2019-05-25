import math
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


class treenode_discrete:
    def __init__(self,feature=None,label=None):
        self.children={}
        self.feature=feature
        self.label=label
        
    def display(self,ind=0):
        if self.children=={}:
            print('   '*ind,str(self.label)+'类')
        else:    
            print('   '*ind,'第'+str(self.feature)+'个属性')
            for value,child in self.children.items():
                print('   '*ind,'为',str(value))
                child.display(ind+1)
            
        
    def predict(self,X_test):
        y_test_pred=[]
        for instance in X_test:
            y_test_pred.append(self.predict_one(instance))
        return np.array(y_test_pred)
            
    def predict_one(self,instance):
        if self.children=={}:
            result=self.label
        else:
            result=self.children[instance[self.feature]].predict_one(instance)
        return result
    
            
def calculate_entropy(X,y):
    y=list(y)
    classunique=list(set(y))
    entropy=0
    for cla in classunique:
        ratio=(y.count(cla))/len(y)
        entropy=entropy-ratio*math.log(ratio,2)
    return entropy


def split_data_discrete(X,y,feature,value):
    sub_X=X[np.where(X[:,feature]==value)]
    sub_y=y[np.where(X[:,feature]==value)]
    return sub_X,sub_y


def choose_best_split_discrete(X,y,feature_set):
    old_entropy=calculate_entropy(X,y)
    #print(old_entropy)
    g=0
    for feature in feature_set:
        values=list(np.unique(X[:,feature]))
        new_entropy=0
        for value in values:
            sub_X,sub_y=split_data_discrete(X,y,feature,value)
            sub_entropy=calculate_entropy(sub_X,sub_y)
            new_entropy+=sub_entropy*len(sub_X)/len(X)
        information_gain=old_entropy-new_entropy
        #print('第',i,'个属性，','划分点为',point,'，划分后熵为',new_entropy,'，信息增益为',information_gain,sep='')
        if information_gain>g:
            g=information_gain
            best_feature=feature
    return best_feature
                

def get_majority_label(y):
    return pd.Series(y).value_counts().index[0]


def create_tree_discrete(X,y,feature_set):
    if len(set(list(y)))==1:
        leaf=treenode_discrete(label=y[0])
        return leaf
    elif len(np.array(pd.DataFrame(X).drop_duplicates()))==1 or feature_set==[]:
        majority_label=get_majority_label(y)
        leaf=treenode_discrete(label=majority_label)
        return leaf
    else:
        best_feature=choose_best_split_discrete(X,y,feature_set)
        tnode=treenode_discrete(feature=best_feature)
        values=list(np.unique(X[:,best_feature]))
        
        for value in values:
            sub_X,sub_y=split_data_discrete(X,y,best_feature,value)
            
            if len(sub_y)==0:
                majority_label=get_majority_label(y)
                leaf=treenode_discrete(label=majority_label)
                tnode.children[value]=leaf
            else:
                sub_feature_set=feature_set.copy()
                sub_feature_set.remove(best_feature)
                tnode.children[value]=create_tree_discrete(sub_X,sub_y,sub_feature_set)
        return tnode


def get_tree(X,y):
    feanum=np.shape(X)[1]
    feature_set=list(range(0,feanum,1))
    tree=create_tree_discrete(X,y,feature_set)
    return tree



data=[]
with open('西瓜数据集2.0.txt','r') as f:
    for line in f.readlines():
        data.append(line.strip().split())
data=np.array(data)
X=data[:,:-1]
y=data[:,-1]
mytree=get_tree(X,y)

X_test=np.array([['青绿', '稍蜷', '清脆', '清晰', '稍凹', '硬滑'],['浅白', '硬挺', '浊响', '清晰', '凹陷', '软粘']])
y_test=np.array(['是','否'])
y_test_pred=mytree.predict(X_test)
print(np.mean(y_test==y_test_pred))
#若去预测['浅白', '硬挺', '浊响', '清晰', '凹陷', '软粘'] 会出问题
#有节点没有弄出所有的分支，只弄出了在他当前下有的value，这样会造成预测时缺少分支
#此问题尚未解决