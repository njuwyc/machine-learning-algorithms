#梯度下降求最小值
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.metrics import roc_auc_score
from math import log,exp



'''
filepath="D:/学习/大三上/机器学习/page-blocks.data"
fr=open(filepath,'r',encoding='utf8').read().split('\n')[0:-1]
print(len(fr))
data=[]
for line in fr:
    line=line.split(' ')
    for i in range(len(line)-1,-1,-1):
        if line[i]=='':
            del line[i]
    data.append(line)
data=pd.DataFrame(data,columns=["height", "lenght", "area", "eccen", "p_black", 
                                "p_and", "mean_tr","blackpix", "blackand", 
                                "wb_trans","block_class"])
print(data)
print(data.iloc[:,-1].value_counts())
X=data.iloc[:,0:-1]
y=data.iloc[:,-1]
'''
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
    label_mapping={'<=50K':1,'>50K':0}
    #调库时这里标记是啥无所谓，但自己的程序要改成0和1，不能是1和2、1和-1
    data['Earnings-Raw']=data['Earnings-Raw'].map(label_mapping)
    #for i in range(feanum):
    #   print(data.iloc[:,i].value_counts())  #转换前的取值样子
    data=data.astype('float')
    for i in range(feanum):
        data.iloc[:,i]=data.iloc[:,i].fillna(data.iloc[:,i].median())
    return data



def sigmoid(x):
    return (float(1)/(1+np.exp(-x)))

def expanddata(dataarr):
    yangnum=np.shape(dataarr)[0]
    datamat=np.column_stack((dataarr,np.ones(yangnum)))
    return datamat
    
def LR_GD(dataarr,labelmat,maxiter,alpha):
    datamat=expanddata(dataarr)
    yangnum,feanum=np.shape(datamat)#这里的feanum是比原来多1列的，每行多了个’1‘
    weight=np.ones(feanum)#即为m+1行的列向量，最后一列即最后想得出的b
    for t in range(maxiter):
        h=sigmoid(np.dot(datamat,weight))
        error=h-labelmat
        gradient=np.dot(datamat.T,error)
        temp=weight
        weight=weight-alpha*gradient
        if np.sum(np.square(weight-temp))<1e-16:
            return weight
    return weight



def LR_SGD(dataarr,labelmat,maxiter,batch_size): #在线算法，有新输入来时可直接做，事先不需知道所有的输入。
    datamat=expanddata(dataarr)
    yangnum,feanum=np.shape(datamat)
    weight=np.ones(feanum)
    for t in range(maxiter):
        index_list=list(range(yangnum))
        np.random.shuffle(index_list)
        batch_number=yangnum//batch_size
        for i in range(batch_number):
            alpha=4/(1.0+t+i)+0.01
            random_index=index_list[i*batch_size:(i+1)*batch_size]
            datamat_batch=datamat[random_index]
            labelmat_batch=labelmat[random_index]
            h=sigmoid(np.dot(datamat_batch,weight))
            error=h-labelmat_batch
            gradient=np.dot(datamat_batch.T,error)
            temp=weight
            weight=weight-alpha*gradient
            if np.sum(np.square(weight-temp))<1e-16:
                return weight
    return weight


'''
import random
def LR_SGD(dataarr,labelmat,maxiter): #在线算法，有新输入来时可直接做，事先不需知道所有的输入。
    datamat=expanddata(dataarr)
    yangnum,feanum=np.shape(datamat)
    weight=np.ones(feanum)
    for t in range(maxiter):
        dataindex=list(range(yangnum))
        for i in range(yangnum): #每次更新只用一个随机的样本，而不是每次都把整个数据集都做一遍
            alpha=4/(1.0+t+i)+0.01  #每次步长有变化
            randomindex=int(random.uniform(0,len(dataindex))) #每次不放回随机拿出一个样本
            h=sigmoid(np.dot(datamat[randomindex],weight))
            error=h-labelmat[randomindex]
            weight=weight-alpha*error*datamat[randomindex]
            del dataindex[randomindex]
    return weight
'''
 

def LR_NEWTON(dataarr,labelmat,maxiter):
    datamat=expanddata(dataarr)
    yangnum,feanum=np.shape(datamat)
    weight=np.ones(feanum)
    for t in range(maxiter):
        h=sigmoid(np.dot(datamat,weight))
        error=h-labelmat
        gradient=np.dot(datamat.T,error)
        r=h*(1-h)
        hesse=np.dot(datamat.T*r,datamat)
        hesse_ni=np.linalg.pinv(hesse)
        temp=weight
        weight=weight-np.dot(hesse_ni,gradient)
    return weight
    
    
    
    
def classify(X,y,weight):
    X_mat=expanddata(X)
    result=sigmoid(np.dot(X_mat,weight))
    classifiedlabel=[]
    for x in result:
        if x>0.5:
            classifiedlabel.append(1)
        else:
            classifiedlabel.append(0)
    classifiedlabel=np.array(classifiedlabel)
    return classifiedlabel
            
        
        
        
'''
X_train=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],
                  [0.556,0.215],[0.403,0.237],[0.481,0.149],[0.437,0.211],
                  [0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],
                  [0.639,0.161],[0.657,0.198],[0.360,0.370],[0.593,0.042],
                  [0.719,0.103]])
y_train=np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,s=40)
'''




from sklearn.datasets.samples_generator import make_blobs
X,y=make_blobs(n_samples=14000,centers=2,random_state=0,cluster_std=1.3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0) #这里训练集75%:测试集25%



fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.scatter(X_train[:,0],X_train[:,1],c=y_train,s=30,cmap='autumn')
ax2 = fig.add_subplot(222)
ax2.scatter(X_test[:,0],X_test[:,1],c=y_test,s=30,cmap='autumn')
ax3 = fig.add_subplot(223)
ax3.scatter(X_train[:,0],X_train[:,1],c=y_train,s=30,cmap='autumn')
ax4 = fig.add_subplot(224)
ax4.scatter(X_test[:,0],X_test[:,1],c=y_test,s=30,cmap='autumn')


'''
#普通梯度下降训练出线性模型
weight=LR_GD(X_train,y_train,1000000,0.01)
#print(weight)

#训练集的分类精度
classified_y_train=classify(X_train,y_train,weight)
trainprecision=np.mean(classified_y_train==y_train)
print(trainprecision)

#画出训练集上的分类决策边界
xfit=np.arange(-3.0,6.0,0.01)
yfit=(-weight[0]*xfit-weight[2])/weight[1]
ax1.plot(xfit,yfit)

#在测试集上预测
classified_y_test=classify(X_test,y_test,weight)
testprecision=np.mean(classified_y_test==y_test)
print(testprecision)
#画出测试集上的分类决策边界，这条线实际上跟训练集图中的是一条线呀
ax2.plot(xfit,yfit)
'''

'''
#随机梯度下降
weight=LR_SGD(X_train,y_train,100000)
#print(weight)

#训练集的分类精度
classified_y_train=classify(X_train,y_train,weight)
trainprecision=np.mean(classified_y_train==y_train)
print(trainprecision)
#画出训练集上的分类决策边界
xfit=np.arange(-3.0,6.0,0.01)
yfit=(-weight[0]*xfit-weight[2])/weight[1]
ax3.plot(xfit,yfit)

#在测试集上预测
classified_y_test=classify(X_test,y_test,weight)
testprecision=np.mean(classified_y_test==y_test)
print(testprecision)
#画出测试集上的分类决策边界，这条线实际上跟训练集图中的是一条线呀
ax4.plot(xfit,yfit)

#普通梯度下降(即每次循环都整体算)收敛慢(三个要估计的参数收敛慢)，而随机梯度下降收敛极快
'''


'''
#adult数据集结果
adult_train_filepath="D:/学习/大三上/机器学习/adult_train.data"
traindata=loaddataset(adult_train_filepath)
X_train=traindata.iloc[:,0:-1]
y_train=traindata.iloc[:,-1]
X_train=sklearn.preprocessing.minmax_scale(X_train)
print(X_train)
weight=LR_GD(X_train,y_train,1000,0.01)
#print(weight)

#训练集的分类精度
classified_y_train=classify(X_train,y_train,weight)
trainprecision=np.mean(classified_y_train==y_train)
print(trainprecision)

#测试集预测精度
adult_test_filepath="D:/学习/大三上/机器学习/adult_test.data"
testdata=loaddataset(adult_test_filepath)
X_test=testdata.iloc[:,0:-1]
y_test=testdata.iloc[:,-1]
X_test=sklearn.preprocessing.minmax_scale(X_test)
classified_y_test=classify(X_test,y_test,weight)
testprecision=np.mean(classified_y_test==y_test)
print(testprecision)
'''

'''
#调库效果
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression().fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
'''





