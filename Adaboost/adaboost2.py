import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from math import log,exp

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


def buildstump(yangnum,feanum,datamat,label,w):    
    stumpbest={}
    stumpbestclass=[]
    e=1
    w=np.array(w)
    numsteps=10
    for i in range(feanum):
        rangemin=datamat[:,i].min()
        rangemax=datamat[:,i].max()
        stepsize=(rangemax-rangemin)/numsteps
        for j in range(-1,numsteps+1):
            for inequal in ['lt','gt']:
                v=rangemin+float(j)*stepsize
                classifiedlabel=stumpclassify(inequal,i,v,datamat)
                errarr=np.array(np.ones(yangnum))
                errarr[classifiedlabel==label]=0
                weightederror=np.dot(w,errarr)
                if weightederror<e:   #通过迭代，每次是不是比上次的好，求出这一轮最好的树桩分类器 
                    stumpbest['fea']=i
                    stumpbest['zhi']=v
                    stumpbest['inequal']=inequal
                    stumpbestclass=classifiedlabel
                    e=weightederror
    return stumpbest,e,stumpbestclass     
            

def sign(current):
    current_classified_label=[]
    for z in current:
        if z<=0:
            current_classified_label.append(-1)
        else:
            current_classified_label.append(1)
    return np.array(current_classified_label)
    
              
def adaboost(datamat,label,yangnum,feanum,T,datamattest,labeltest,yangnumtest,feanumtest):
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
    current_test_precision_all_time=[]
    current_test_auc_all_time=[]
    for t in range(T):
        stumpbest,weightederror,stumpbestclass=buildstump(yangnum,feanum,datamat,label,w)
        print("第",t+1,"轮得到的最优弱分类器:",stumpbest,"其加权错误率为:",weightederror)
        if weightederror>0.5:
            break
        
        stumpbest_all_time.append(stumpbest)
        
        stumpbestclass=np.array(stumpbestclass)
        
        weightederror_all_time.append(weightederror)
        
        alpha=0.5*(log((1.0-weightederror)/weightederror))
        alpha_all_time.append(alpha)
        stumpbest['alpha']=alpha
        
        expon=np.multiply(-1.0*alpha*label,stumpbestclass)
        w=np.multiply(w,np.exp(expon))
        w=w/w.sum()
        #print("第",t+1,"轮结束后的样本权值分布:",w)
        
        current=current+alpha*stumpbestclass
        #print(current)
        current_classified_label=sign(current)
        #print(current_classified_label)
        current_precision=np.mean(current_classified_label==label)
        print("集成",t+1,"个最优弱分类器后训练集上的分类精度为:",current_precision)
        current_precision_all_time.append(current_precision)
        current_auc=roc_auc_score(label,current_classified_label)
        print("集成",t+1,"个最优弱分类器后训练集上的auc值为:",current_auc)
        current_auc_all_time.append(current_auc)
        
        current_classified_label_test=adaboost_test(datamattest,stumpbest_all_time,yangnumtest,feanumtest)
        current_test_precision=np.mean(current_classified_label_test==labeltest)
        print("集成",t+1,"个最优弱分类器后测试集上的分类预测精度为:",current_test_precision)
        current_test_precision_all_time.append(current_test_precision)
        current_test_auc=roc_auc_score(labeltest,current_classified_label_test)
        print("集成",t+1,"个最优弱分类器后测试集上的auc值为:",current_test_auc)
        current_test_auc_all_time.append(current_test_auc)
        
        if current_precision==1.0:
            break
        
    print("每轮的最优弱分类器:",stumpbest_all_time)
    #print("每轮分类器的加权错误率:",weightederror_all_time)  
    #print("每轮分类器的alpha系数:",alpha_all_time)      
    print("集成的分类器每次增加1个的每次训练集上的精度:",current_precision_all_time)
    print("集成的分类器每次增加1个的每次训练集上的auc值:",current_auc_all_time)
    print("集成的分类器每次增加1个的每次测试集上的精度:",current_test_precision_all_time)
    print("集成的分类器每次增加1个的每次测试集上的auc值:",current_test_auc_all_time)
    current_precision_all_time=pd.Series(current_precision_all_time)
    current_precision_all_time.plot()
    #current_auc_all_time=pd.Series(current_auc_all_time)
    #current_auc_all_time.plot()
    current_test_precision_all_time=pd.Series(current_test_precision_all_time)
    current_test_precision_all_time.plot()
    #current_test_auc_all_time=pd.Series(current_test_auc_all_time)
    #current_test_auc_all_time.plot()
    #return stumpbest_all_time




def adaboost_test(datamattest,stumpbest_all_time,yangnumtest,feanumtest):
    classifiedlabel=[]
    test_final=np.array(np.zeros(yangnumtest))
    #num=1
    for stumpbest in stumpbest_all_time:
        i=stumpbest['fea']
        v=stumpbest['zhi']
        inequal=stumpbest['inequal']
        classifiedlabel=np.array(stumpclassify(inequal,i,v,datamattest))
        test_final=test_final+stumpbest['alpha']*classifiedlabel
        #print("当前已加权累加了",num,"个最优弱分类器的预测结果")
        #num+=1
    #print("测试集最终分类预测的数值:",test_final)
    final_classified_label_test=sign(test_final)
    #print("测试集最终分类预测的结果:",final_classified_label_test)
    return final_classified_label_test
        
        
        
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
data=loaddataset(adult_train_filepath)
#data.to_csv("D:/学习/大三上/机器学习/adult__preprocessed_data.csv")
#data=pd.DataFrame(pd.read_csv(open("D:/学习/大三上/机器学习/adult__preprocessed_data.csv")))
#print(np.shape(data))
datamat=data.iloc[:,0:-1]
label=data.iloc[:,-1]
#print(label.value_counts())
datamat=np.array(datamat)
label=np.array(label)
yangnum,feanum=np.shape(datamat)
print("训练集的样本数量和特征数分别为:",yangnum,feanum)

adult_test_filepath="D:/学习/大三上/机器学习/adult_test.data"
datatest=loaddataset(adult_test_filepath)
datamattest=datatest.iloc[:,0:-1]
labeltest=datatest.iloc[:,-1]
datamattest=np.array(datamattest)
labeltest=np.array(labeltest)
yangnumtest,feanumtest=np.shape(datamattest)
print("测试集的样本数量和特征数分别为:",yangnumtest,feanumtest)


adaboost(datamat,label,yangnum,feanum,10,datamattest,labeltest,yangnumtest,feanumtest)



final_classified_label_test=adaboost_test(datamattest,stumpbest_all_time,yangnumtest,feanumtest)
test_precision=np.mean(final_classified_label_test==labeltest)
print("测试集上精度为:",test_precision)


'''
#调库
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=10)
model=clf.fit(datamat, label)
y_pred = model.predict(datamattest)
precision=np.mean(y_pred==labeltest)
print(precision)
current_auc=roc_auc_score(labeltest,y_pred)
print(current_auc)
'''
#逐个的结果
#model.staged_predict(X_test)




