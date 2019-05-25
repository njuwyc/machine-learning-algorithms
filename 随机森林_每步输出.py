import numpy as np
import pandas as pd
from math import log,exp
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def loaddataset(filepath):
    data=[]
    fr=open(filepath,'r',encoding='utf8').read().split('\n')[0:-2]
    #print(fr)
    feanum=len(fr[0].split(' ')) #属性数量加最后一列label
    yangnum=len(fr)
    for line in fr:
        line=line.split(', ')
        line[-1]=line[-1].replace('.','')
        data.append(line)
    for i in range(yangnum):
        for j in range(feanum):
            if data[i][j]=='?':
                data[i][j]=np.NaN
    data=pd.DataFrame(data,columns=["Age", "Work-Class", "fnlwgt", 
                                    "Education", "Education-Num", 
                                    "Marital-Status", "Occupation",
                                    "Relationship", "Race", "Sex", 
                                    "Capital-gain","Capital-loss", 
                                    "Hours-per-week", "Native-Country",
                                    "Earnings-Raw"])
    #print(data['Work-Class'].value_counts())
    #print(data.iloc[:,1].value_counts())
    #print(np.shape(data)) 
    #total = data.isnull().sum().sort_values(ascending=False)
    #print(total)  #每列有多少na的
    #data=data.dropna()   #如果直接去除有na的样本
    #print(np.shape(data)) #如果去除有na的行，之后
    #for i in range(feanum):
    #   print(data.iloc[:,i].value_counts()) #转换前的取值样子
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
    label_mapping={'<=50K':1,'>50K':-1}
    data['Earnings-Raw']=data['Earnings-Raw'].map(label_mapping)
    #for i in range(feanum):
    #   print(data.iloc[:,i].value_counts())  #转换前的取值样子
    data=data.astype('float')
    for i in range(feanum):
        data.iloc[:,i]=data.iloc[:,i].fillna(data.iloc[:,i].median())
    return data


def bootstrap(X,y):
    m=np.shape(y)[0]
    rindex=[]
    for i in range(m):
        r=random.randint(0,m-1)
        rindex.append(r)
    return X[rindex],y[rindex]


def sign(arr):
    result=np.zeros([len(arr)])
    result[arr<=0]=-1
    result[arr>0]=1
    return result


def ensemble(preds):
    preds=np.array(preds)
    m=np.shape(preds)[1]
    result=np.zeros([m])
    for pred in preds:
        result+=pred
    return sign(result)


def create_randomforest(X_train_all,y_train_all,T,k):
    skf = StratifiedKFold (n_splits=k)
    X_cv=[]
    y_cv=[]
    for train_index,valid_index in skf.split(X_train_all,y_train_all):
        X_cv.append([X_train_all[train_index],X_train_all[valid_index]])
        y_cv.append([y_train_all[train_index],y_train_all[valid_index]])
    y_valid_pred_all=[[] for i in range(k)]
    auc_all=[[] for t in range(T)]
    auc_average=[]
    for t in range(0,T,1):
        #print('第',t+1,'棵决策树')
        for i in range(k):
            X_train=X_cv[i][0]
            y_train=y_cv[i][0]
            X_valid=X_cv[i][1]
            y_valid=y_cv[i][1]
            
            X_bs,y_bs=bootstrap(X_train,y_train)
            clf=DecisionTreeClassifier(max_features='log2').fit(X_bs,y_bs)
            
            y_valid_pred=clf.predict(X_valid)
            
            y_valid_pred_all[i].append(y_valid_pred)
            auc_all[t].append(roc_auc_score(y_valid,ensemble(y_valid_pred_all[i])))
        
        if t%5==0:
            print('第',t+1,'个',np.array(auc_all[t]).mean())
            auc_average.append(np.array(auc_all[t]).mean())
    return auc_average





'''
datamat=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],
                  [0.556,0.215],[0.403,0.237],[0.481,0.149],[0.437,0.211],
                  [0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],
                  [0.639,0.161],[0.657,0.198],[0.360,0.370],[0.593,0.042],
                  [0.719,0.103]])
labelmat=np.array([1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2])
label=np.array([1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
yangnum,feanum=np.shape(datamat)
'''
'''
datamat=np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
label=np.array([1,1,1,-1,-1,-1,1,1,1,-1])
yangnum,feanum=np.shape(datamat)
'''
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datamat[:,0],datamat[:,1],20*labelmat,20*labelmat)
'''
'''
datamat=np.array([[1.0,2.1],[2.0,1.1],[1.3,1.0],[1.0,1.0],[2.0,1.0]])
label=np.array([1,1,-1,-1,1])
yangnum,feanum=np.shape(datamat)
'''

adult_train_filepath="D:/学习/大三上/机器学习/adult_train.data"
traindata=loaddataset(adult_train_filepath)


X_train=np.array(traindata.iloc[:,0:-1])
y_train=np.array(traindata.iloc[:,-1])
auc_average=create_randomforest(X_train,y_train,500,5)
