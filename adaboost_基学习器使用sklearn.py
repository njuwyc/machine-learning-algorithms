import numpy as np
import pandas as pd
from math import log
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def loaddataset(filepath):
    data=[]
    with open(filepath,'r') as f:
        for line in f.readlines()[0:-1]:
            line=line.strip()
            if line[-1]=='.':
                line=line[:-1]
            line=line.split(', ')
            if line[-1]=='?':
                continue
            for i in range(len(line)):
                if line[i]=='?':
                    line[i]=np.NaN
            data.append(line)
    data=pd.DataFrame(data,columns=["Age", "Work-Class", "fnlwgt", 
                                    "Education", "Education-Num", 
                                    "Marital-Status", "Occupation",
                                    "Relationship", "Race", "Sex", 
                                    "Capital-gain","Capital-loss", 
                                    "Hours-per-week", "Native-Country",
                                    "Earnings-Raw"])
    data['Earnings-Raw'][data['Earnings-Raw']=='<=50K']=1
    data['Earnings-Raw'][data['Earnings-Raw']=='>50K']=-1
    for i in [1,5,6,7,8,9,13]:
        mapping = {label:idx for idx,label in enumerate(set(data.iloc[:,i]))}
        if np.nan in mapping.keys():
            del mapping[np.NaN]
        data.iloc[:,i]=data.iloc[:,i].map(mapping)
    Education_mapping={'Preschool':1,'1st-4th':2,'5th-6th':3,'7th-8th':4,
                       '9th':5,'10th':6,'11th':7,'12th':8,'HS-grad':9,
                       'Some-college':10,'Assoc-voc':11,'Assoc-acdm':12,
                       'Bachelors':13,'Masters':14,'Prof-school':15,
                       'Doctorate':16}
    data['Education']=data['Education'].map(Education_mapping)
    data=data.astype('float')
    data=data.fillna(data.median())
    return data


def adaboost_train(X,y,T):
    m=np.shape(X)[0]
    w=np.full([m],1/m)
    model={}
    alpha_all=[]
    stump_all=[]
    for t in range(T):
        stump=DecisionTreeClassifier(max_depth=1)
        stump=stump.fit(X,y,sample_weight=w)
        stump_all.append(stump)
        
        y_pred=stump.predict(X)
        weighted_error=1-stump.score(X,y,sample_weight=w)
        if weighted_error>0.5:
            break
        alpha=0.5*(log((1.0-weighted_error)/weighted_error))
        alpha_all.append(alpha)
        
        w=w*np.exp(-1.0*alpha*y*y_pred)
        w=w/w.sum()
        
    model['stump_all']=stump_all
    model['alpha_all']=alpha_all
    return model
        

def sign(arr):
    result=np.zeros([len(arr)])
    result[arr<=0]=-1
    result[arr>0]=1
    return result
    

def adaboost_predict(model,X):
    T=len(model['stump_all'])
    m=np.shape(X)[0]
    pred=np.zeros([m])
    for i in range(T):
        stump=model['stump_all'][i]
        alpha=model['alpha_all'][i]
        pred+=alpha*stump.predict(X)
    return sign(pred)
    
    
    
adult_train_filepath="D:/学习/大三上/机器学习/adult_train.data"
traindata=loaddataset(adult_train_filepath)
adult_test_filepath="D:/学习/大三上/机器学习/adult_test.data"
testdata=loaddataset(adult_test_filepath)
X_train_all=np.array(traindata.iloc[:,0:-1])
y_train_all=np.array(traindata.iloc[:,-1])
X_test=np.array(testdata.iloc[:,0:-1])
y_test=np.array(testdata.iloc[:,-1])


T=500
k=5
auc_all=[]
auc_best=0
t_best=0
for t in range(1,T+2,5):
    auc=0
    skf = StratifiedKFold(n_splits=k)
    for train_index,valid_index in skf.split(X_train_all,y_train_all):
        #print(train_index,valid_index)
        X_train=X_train_all[train_index]
        y_train=y_train_all[train_index]
        X_valid=X_train_all[valid_index]
        y_valid=y_train_all[valid_index]
        model=adaboost_train(X_train,y_train,T=t)
        y_valid_pred=adaboost_predict(model,X_valid)
        auc+=roc_auc_score(y_valid,y_valid_pred)
    auc=auc/k
    print(t,'个基学习器',auc)
    auc_all.append(auc)
    if auc>auc_best:
        auc_best=auc
        t_best=t

model_best=adaboost_train(X_train_all,y_train_all,T=t_best)
y_test_pred=adaboost_predict(model_best,X_test)
test_auc=roc_auc_score(y_test,y_test_pred)


'''
from sklearn.ensemble import AdaBoostClassifier
T=500
k=5
auc_all=[]
auc_best=0
t_best=0
for t in range(1,T+2,5):
    auc=0
    skf = StratifiedKFold (n_splits=k)
    for train_index,valid_index in skf.split(X_train_all,y_train_all):
        X_train=X_train_all[train_index]
        y_train=y_train_all[train_index]
        X_valid=X_train_all[valid_index]
        y_valid=y_train_all[valid_index]
        ad=AdaBoostClassifier(n_estimators=t,algorithm='SAMME').fit(X_train,y_train)
        y_valid_pred=ad.predict(X_valid)
        auc+=roc_auc_score(y_valid,y_valid_pred)
    auc=auc/k
    print(t,'个基学习器',auc)
    auc_all.append(auc)
    if auc>auc_best:
        auc_best=auc
        t_best=t
'''  




