import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from math import log,exp

def loaddataset(filepath):
    data=[]
    fr=open(filepath,'r',encoding='utf8').read().split('\n')[0:-2]
    feanum=len(fr[0].split(' ')) #属性数量加最后一列label
    yangnum=len(fr)
    for line in fr:
        line=line.split(', ')
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
    print(np.shape(data)) #去除有na的行前
    #total = data.isnull().sum().sort_values(ascending=False)
    #print(total)  #每列有多少na的
    data=data.dropna()   
    print(np.shape(data)) #去除有na的行后
    #for i in range(feanum):
    #   print(data.iloc[:,i].value_counts()) #转换前的取值样子
    for i in [1,5,6,7,8,9,13]:
        mapping = {label:idx for idx,label in enumerate(set(data.iloc[:,i]))}
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
    return data




def get_dividepoints(datamat):
    datamat_t=datamat.T.copy() #可变数据类型 一定要注意复制一份 不然原来的也变了
    fea_mean=[]
    for feaxiang in datamat_t:
        feaxiang.sort()
        feaxiang_mean=[]
        for i in range(len(feaxiang)-1):
            feaxiang_mean.append(0.5*(feaxiang[i]+feaxiang[i+1]))
        feaxiang_mean=list(set(feaxiang_mean))
        feaxiang_mean.sort()
        fea_mean.append(feaxiang_mean)
    return fea_mean


def stumpclassify(inequal,i,v,datamat):
    classifiedlabel=[]
    for j in range(len(datamat)):
        if inequal=='lt':  #小于等于该值是负类，大于该值为正类
            if datamat[j][i]<=v:
                classifiedlabel.append(-1)
            else:
                classifiedlabel.append(1)
        else:              #小于等于该值是正类，大于该值为负类
            if datamat[j][i]<=v:
                classifiedlabel.append(1)
            else:
                classifiedlabel.append(-1)
    return classifiedlabel


def buildstump(feanum,yangnum,fea_mean,datamat,label,w):
    #用一个列表存储这一轮所有的树桩分类器        
    stumpbest={}
    stumpbestclass=[]
    e=1
    w=np.array(w)
    for i in range(feanum):
        for v in fea_mean[i]:
            for inequal in ['lt','gt']:
                classifiedlabel=stumpclassify(inequal,i,v,datamat)
                errarr=np.array(np.ones(yangnum))
                errarr[classifiedlabel==label]=0
                weightederror=np.dot(w,errarr)
                if weightederror<e:   #通过迭代，每次是不是比上次的好，求出这一轮最好的树桩分类器 
                    stumpbest['fea']=i
                    stumpbest['zhi']=v
                    stumpbest['inequal']=inequal
                    stumpbest['error']=weightederror
                    stumpbestclass=classifiedlabel
                    e=weightederror
    return stumpbest,stumpbest['error'],stumpbestclass     
            

def sign(current):
    current_classified_label=[]
    for z in current:
        if z<=0:
            current_classified_label.append(-1)
        else:
            current_classified_label.append(1)
    return np.array(current_classified_label)
    
              
def adaboost(datamat,label,feanum,yangnum,T):
    w=[]
    for i in range(yangnum):
        w.append(1.0/yangnum)
    #print("开始时的样本权值分布:",w)
    w=np.array(w)
    stumpbest_all_time=[]
    weightederror_all_time=[]
    alpha_all_time=[]
    current=np.array(np.zeros(yangnum))
    current_precision_all_time=[]
    current_auc_all_time=[]
    fea_mean=get_dividepoints(datamat) 
    #print(fea_mean)         
    for t in range(T):
        stumpbest,weightederror,stumpbestclass=buildstump(feanum,yangnum,fea_mean,datamat,label,w)
        print("第",t+1,"轮得到的最优弱分类器:",stumpbest,weightederror)
        if weightederror>0.5:
            break
        stumpbest_all_time.append(stumpbest)
        stumpbestclass=np.array(stumpbestclass)
        weightederror_all_time.append(weightederror)
        alpha=0.5*(log((1.0-weightederror)/weightederror))
        alpha_all_time.append(alpha)
        expon=np.multiply(-1.0*alpha*label,stumpbestclass)
        w=np.multiply(w,np.exp(expon))
        w=w/w.sum()
        #print("第",t+1,"轮结束后的样本权值分布:",w)
        current=current+alpha*stumpbestclass
        current_classified_label=sign(current)
        current_precision=np.mean(current_classified_label==label)
        print("集成",t+1,"个最优弱分类器后的分类精度为:",current_precision)
        current_precision_all_time.append(current_precision)
        current_auc=roc_auc_score(label,current_classified_label)
        print("集成",t+1,"个最优弱分类器后的auc值为:",current_auc)
        current_auc_all_time.append(current_auc)
        if current_precision==1.0:
            break
    print("每轮的最优弱分类器:",stumpbest_all_time)
    print("每轮分类器的加权错误率:",weightederror_all_time)  
    print("每轮分类器的alpha系数:",alpha_all_time)      
    print("集成的分类器每次增加1个的每次精度:",current_precision_all_time)
    print("集成的分类器每次增加1个的每次auc值:",current_auc_all_time)
    current_precision_all_time=pd.Series(current_precision_all_time,index=np.arange(1,21,1))
    current_precision_all_time.plot()
    current_auc_all_time=pd.Series(current_auc_all_time,index=np.arange(1,21,1))
    current_auc_all_time.plot()
'''
datamat=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],
                  [0.556,0.215],[0.403,0.237],[0.481,0.149],[0.437,0.211],
                  [0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],
                  [0.639,0.161],[0.657,0.198],[0.360,0.370],[0.593,0.042],
                  [0.719,0.103]])
labelmat=np.array([1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2])
label=np.array([1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
'''
'''
datamat=np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
label=np.array([1,1,1,-1,-1,-1,1,1,1,-1])
'''
'''
yangnum,feanum=np.shape(datamat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datamat[:,0],datamat[:,1],c=y_test,s=30,cmap='autumn')
'''


adult_train_filepath="D:/学习/大三上/机器学习/adult_train.data"
data=loaddataset(adult_train_filepath)
datamat=data.iloc[:,0:-1]
label=data.iloc[:,-1]
#print(label.value_counts())
datamat=np.array(datamat)
label=np.array(label)
yangnum,feanum=np.shape(datamat)
print(yangnum,feanum)

adaboost(datamat,label,feanum,yangnum,1)


 
  


    
        





































