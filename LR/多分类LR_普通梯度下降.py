import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.metrics import roc_auc_score
from math import log,exp
from sklearn.model_selection import train_test_split


def loaddataset(filepath):
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
    data=data.astype('float')
    #print(data)
    #print(data.iloc[:,-1].value_counts())
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
        weight=weight-alpha*gradient
    return weight


def multi_classify(X,y,weight_all_time):
    pp_all_time={}
    X_mat=expanddata(X)
    for label in weight_all_time.keys():
        weight_current=weight_all_time[label]
        pp_current=sigmoid(np.dot(X_mat,weight_current))
        pp_all_time[label]=pp_current
    pp_all_time=pd.DataFrame(pp_all_time)
    classified_y=np.array(pp_all_time.idxmax(axis=1))
    return classified_y


def LR_fit(X,y,maxiter,alpha):
    yset=list(set(y))
    #print(yset)
    weight_all_time={}
    for label in yset:
        print(yset.index(label)+1,sep='',end=' ')
        y_current=np.ones(len(y))
        for i in range(len(y)):
            if y[i]!=label:
                y_current[i]=0
        #print(y_current)
        weight_current=LR_GD(X,y_current,maxiter,alpha)
        weight_all_time[label]=weight_current
    return weight_all_time

def get_micro_p(confusion_matrix):
    p=0
    for i in range(len(confusion_matrix)):
        p+=confusion_matrix[i][i]
    micro_p=p/np.sum(confusion_matrix)
    return micro_p

def get_micro_r(confusion_matrix):
    r=0
    for i in range(len(confusion_matrix)):
        r+=confusion_matrix[i][i]
    micro_r=r/np.sum(confusion_matrix)
    return micro_r

def get_micro_f1(micro_p,micro_r):
    return (2*micro_p*micro_r)/(micro_p+micro_r)
    
def get_macro_p(confusion_matrix):
    p=0
    for i in range(len(confusion_matrix)):
        p+=confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
    macro_p=p/len(confusion_matrix)
    return macro_p

def get_macro_r(confusion_matrix):
    r=0
    for i in range(len(confusion_matrix)):
        r+=confusion_matrix[i][i]/np.sum(confusion_matrix[i,:])
    macro_r=r/len(confusion_matrix)
    return macro_r

def get_macro_f1(confusion_matrix):
    f1=0
    for i in range(len(confusion_matrix)):
        p=confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
        r=confusion_matrix[i][i]/np.sum(confusion_matrix[i,:])
        f1+=2*p*r/(p+r)
    macro_f1=f1/len(confusion_matrix)
    return macro_f1
    
'''
from sklearn.datasets.samples_generator import make_blobs
X,y=make_blobs(n_samples=1000,centers=3,random_state=0,cluster_std=1.05)
#plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap='autumn')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0) #这里训练集75%:测试集25%
'''


traindata=pd.read_csv(open("D:/学习/大三下/机器学习导论2019/ML2019-PS2-dataset/train_set.csv"))
X_train=traindata.iloc[:,0:-1]
y_train=traindata.iloc[:,-1]
X_train=sklearn.preprocessing.minmax_scale(X_train)
testdata=pd.read_csv(open("D:/学习/大三下/机器学习导论2019/ML2019-PS2-dataset/test_set.csv"))
X_test=testdata.iloc[:,0:-1]
y_test=testdata.iloc[:,-1]
X_test=sklearn.preprocessing.minmax_scale(X_test)


'''
best_test_accuracy=0
best_model=[]
best_classified_y_test=[]
train_accuracy_all_figure=[]
test_accuracy_all_figure=[]
maxiter_all_figure=[]
alpha_list=[0.0001,0.0003,0.001,0.003,0.01,0.03,0.1]
for alpha in alpha_list:
    train_accuracy_one_figure=[]
    test_accuracy_one_figure=[]
    for maxiter in range(100,1001,100):
        print('学习率为',alpha,'最大迭代次数为',maxiter,'时：')
        model=LR_fit(X_train,y_train,maxiter=maxiter,alpha=alpha)
        
        classified_y_train=multi_classify(X_train,y_train,model)
        train_accuracy=np.mean(y_train==classified_y_train)
        print('这时训练集精度为',train_accuracy)
        
        classified_y_test=multi_classify(X_test,y_test,model)
        test_accuracy=np.mean(y_test==classified_y_test)
        print('测试集精度为',test_accuracy)
        
        if test_accuracy>best_test_accuracy:
            best_test_accuracy=test_accuracy
            best_model=[alpha,maxiter,train_accuracy,test_accuracy]
            best_classified_y_test=classified_y_test
        
        train_accuracy_one_figure.append(train_accuracy)
        test_accuracy_one_figure.append(test_accuracy)
    
    train_accuracy_all_figure.append(train_accuracy_one_figure)
    test_accuracy_all_figure.append(test_accuracy_one_figure)
    maxiter_all_figure.append(list(range(100,1001,100)))


fig_num=0
for i in range(len(alpha_list)):
    fig = plt.figure()
    axes=fig.add_subplot(111)
    axes.set_title('Learning rate='+str(alpha_list[i]))
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Accuracy')
    axes.set_xticks(maxiter_all_figure[fig_num])
    axes.plot(np.array(maxiter_all_figure[fig_num]),np.array(train_accuracy_all_figure[fig_num]),label='train')
    axes.plot(np.array(maxiter_all_figure[fig_num]),np.array(test_accuracy_all_figure[fig_num]),label='test')
    axes.legend(loc='best')
    fig_num+=1
'''

best_test_accuracy=0
alpha_best=0.001
train_accuracy_all_time=[]
test_accuracy_all_time=[]
for maxiter in range(100,10101,500):
    print('最大迭代次数为',maxiter,'时：')
    model=LR_fit(X_train,y_train,maxiter=maxiter,alpha=alpha_best)
    
    classified_y_train=multi_classify(X_train,y_train,model)
    train_accuracy=np.mean(y_train==classified_y_train)
    print('这时训练集精度为',train_accuracy)
    
    classified_y_test=multi_classify(X_test,y_test,model)
    test_accuracy=np.mean(y_test==classified_y_test)
    print('测试集精度为',test_accuracy)
    
    if test_accuracy>best_test_accuracy:
        best_test_accuracy=test_accuracy
        best_model=[maxiter,train_accuracy,test_accuracy]
        best_classified_y_test=classified_y_test
    
    train_accuracy_all_time.append(train_accuracy)
    test_accuracy_all_time.append(test_accuracy)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('Best learning rate:0.001')
ax.set_xlabel('Iteration')
ax.set_ylabel('Accuracy')
ax.set_xticks(range(100,10101,1000))
ax.plot(np.array(range(100,10101,500)),np.array(train_accuracy_all_time),label='train')
ax.plot(np.array(range(100,10101,500)),np.array(test_accuracy_all_time),label='test')
ax.legend(loc='best')

print('最佳超参数组合及对应训练集精度和测试集精度:',best_model)

confusion_matrix=[[0 for i in range(len(list(set(y_test))))] for i in range(len(list(set(y_test))))]
for i in range(len(y_test)):
    confusion_matrix[y_test[i]-1][best_classified_y_test[i]-1]+=1
confusion_matrix=np.array(confusion_matrix)
print(confusion_matrix)

micro_p=get_micro_p(confusion_matrix)
micro_r=get_micro_r(confusion_matrix)
micro_f1=get_micro_f1(micro_p,micro_r)
macro_p=get_macro_p(confusion_matrix)
macro_r=get_macro_r(confusion_matrix)
macro_f1=get_macro_f1(confusion_matrix)
print(micro_p,micro_r,micro_f1,macro_p,macro_r,macro_f1,sep='\n')




'''
yset=list(set(y_train))
print(yset)
weight_all_time=[]
pp_all_time=[]
X_train_mat=expanddata(X_train)
for label in yset:
    print('第',yset.index(label)+1,'个分类器')
    y_current=np.ones(len(y_train))
    for i in range(len(y_train)):
        if y_train[i]!=label:
            y_current[i]=0
    #print(y_current)
    weight_current=LR_GD(X_train,y_current,8000,0.01)
    weight_all_time.append(weight_current)
    pp_current=sigmoid(np.dot(X_train_mat,weight_current))
    pp_all_time.append(pp_current)
pp_all_time=np.array(pp_all_time) 
classified_y_train=[]
best_pp_axis=pp_all_time.argmax(axis=0)
for j in best_pp_axis:
    classified_y_train.append(yset[j])
classified_y_train=np.array(classified_y_train)
print(classified_y_train)
print(np.mean(y_train==classified_y_train))
'''

'''
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(max_iter=500).fit(X_train,y_train)
classified_y_train=clf.predict(X_train)
print(np.mean(classified_y_train==y_train))
classified_y_test=clf.predict(X_test)
print(np.mean(classified_y_test==y_test))
'''