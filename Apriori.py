'''
import time
def get_C1(data):
    C1=[]
    for tran in data:
        for item in tran:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    C1=list(map(set,C1))
    return C1


def get_sup(itemset,D):
    sup=0
    for transet in D:
        if itemset.issubset(transet)==True:
            sup+=1
    sup=sup/len(D)
    return sup
        

def get_C(L,k):
    C=[]
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            if sorted(list(L[i]))[:k-2]==sorted(list(L[j]))[:k-2]:
                C.append(L[i] | L[j])
    return C


def get_L(C,D,minsup):
    L=[]
    for itemset in C:
        sup=get_sup(itemset,D)
        if sup>=minsup:
            L.append(itemset)
    return L


def get_freqsets(data,minsup):
    freqsets=[]
    C1=get_C1(data) #C记录“这一次”的候选项集
    D=list(map(set,data)) #D记录事件列表的集合形式，便于查看是否子集、求支持度
    L=get_L(C1,D,minsup) #L记录“这一次“的频繁项集
    freqsets.extend(L)
    k=2
    while L!=[]:  #L记录“这一次“的频繁项集
        C=get_C(L,k) #C记录“这一次”的候选项集
        #print(C)
        L=get_L(C,D,minsup)
        freqsets.extend(L)
        k+=1
    return freqsets


def get_conf(a,b,D):
    return get_sup(a,D)/get_sup(a-b,D)
    

def get_N(M,k):
    N=[]
    for i in range(len(M)):
        for j in range(i+1,len(M)):
            if sorted(list(M[i]))[:k-2]==sorted(list(M[j]))[:k-2]:
                N.append(M[i] | M[j])
    return N
    

def get_rules_from_one_freqset(D,freqset,minconf): #N类似于C，M类似于L，一开始的freqset类似于C1即N1
    rules_one_freqset=[]
    M=[]
    for item in freqset: #item为后件，准确说是set([item])是后件
        conf=get_conf(freqset,set([item]),D)
        if conf>=minconf:
            rules_one_freqset.append((freqset-set([item]),set([item]),conf))
            M.append(set([item]))
    k=2
    while M!=[]:
        N=get_N(M,k)
        if (len(N)==1 and N[0]==freqset) or len(N)==0:
            break  #若N长度为1且这项就是freqset，则会有空集的支持度计数并可能会出现空集-->N中那个项集的规则，这是不需要的
        M=[]
        for itemset in N: #itemset是后件
            conf=get_conf(freqset,itemset,D) #itemset是后件
            #print(conf)
            if conf>=minconf:
                rules_one_freqset.append((freqset-itemset,itemset,conf))
                M.append(itemset)
        k+=1
    return rules_one_freqset
    
            
def get_rules(D,freqsets,minconf):
    rules=[]
    for freqset in freqsets:
        if len(freqset)>1:
            rules_one_freqset=get_rules_from_one_freqset(D,freqset,minconf)
            rules.extend(rules_one_freqset)
    return rules

    
def Apriori(data,minsup,minconf): 
    D=list(map(set,data)) #D记录事件列表的集合形式，便于查看是否子集、求支持度
    freqsets=get_freqsets(D,minsup)
    rules=get_rules(D,freqsets,minconf)
    return freqsets,rules


def load_data(filepath):
    data=[]
    with open(filepath,'r',encoding='utf-8-sig') as f:
        lines=f.readlines()
        for line in lines:
            data.append(line.strip().split(','))
    return data
    

if __name__=='__main__':
    time1=time.time()
    
    data=load_data('D:/学习/大三下/数据仓库与数据挖掘/guanlian_new.csv')
    minsup=0.3
    minconf=0.9 #可修改为别的值
    freqsets,rules=Apriori(data,minsup,minconf)
    rules.sort(key=lambda x:x[2],reverse=True)
    print(len(freqsets))
    print(len(rules))
    
    time2=time.time()
    print(time2-time1)
'''






#改freqsets为字典，把频繁项集（frozenset不可变集合）当作freqsets的键，记录每个频繁项集的支持度
#之后求置信度时由于频繁项集的子集一定是频繁项集，
#所以之后求置信度时需要的所有支持度数值都已经保存在freqsets字典当中了，
#字典相当于哈希表，去找一下即可，不需要重新求支持度来计算置信度
import time
def get_C1(data):
    C1=[]
    for tran in data:
        for item in tran:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    C1=list(map(set,C1))
    return C1


def get_sup(itemset,D):
    sup=0
    for transet in D:
        if itemset.issubset(transet)==True:
            sup+=1
    sup=sup/len(D)
    return sup
        

def get_C(L,k):
    C=[]
    fs=list(L.keys())
    for i in range(len(fs)):
        for j in range(i+1,len(fs)):
            if sorted(list(fs[i]))[:k-2]==sorted(list(fs[j]))[:k-2]:
                C.append(fs[i] | fs[j])
    return C


def get_L(C,D):
    L={}
    for itemset in C:
        sup=get_sup(itemset,D)
        if sup>=minsup:
            L[frozenset(itemset)]=sup
    return L


def get_freqsets(data,minsup):
    freqsets={}
    C1=get_C1(data) #C记录“这一次”的候选项集
    D=list(map(set,data)) #D记录事件列表的集合形式，便于查看是否子集、求支持度
    L=get_L(C1,D) #L记录“这一次“的频繁项集
    freqsets.update(L)
    k=2
    while L!={}:  #L记录“这一次“的频繁项集
        C=get_C(L,k) #C记录“这一次”的候选项集
        #print(C)
        L=get_L(C,D)
        freqsets.update(L)
        k+=1
    return freqsets


def get_conf(a,b,freqsets):
    return freqsets[frozenset(a)]/freqsets[frozenset(a-b)]
    

def get_N(M,k):
    N=[]
    for i in range(len(M)):
        for j in range(i+1,len(M)):
            if sorted(list(M[i]))[:k-2]==sorted(list(M[j]))[:k-2]:
                N.append(M[i] | M[j])
    return N
    

def get_rules_from_one_freqset(freqsets,freqset,minconf): #N类似于C，M类似于L，一开始的freqset类似于C1即N1
    rules_one_freqset=[]
    M=[]
    for item in freqset: #item为后件，准确说是set([item])是后件
        conf=get_conf(freqset,set([item]),freqsets)
        if conf>=minconf:
            rules_one_freqset.append((set(freqset)-set([item]),set([item]),conf))
            M.append(set([item]))
    k=2
    while M!=[]:
        N=get_N(M,k)
        if (len(N)==1 and N[0]==set(freqset)) or len(N)==0:
            break  #若N长度为1且这项就是freqset，则会有空集的支持度计数并可能会出现空集-->N中那个项集的规则，这是不需要的
        M=[]
        for itemset in N: #itemset是后件
            conf=get_conf(freqset,itemset,freqsets) #itemset是后件
            #print(conf)
            if conf>=minconf:
                rules_one_freqset.append((set(freqset)-set(itemset),set(itemset),conf))
                M.append(set(itemset))
        k+=1
    return rules_one_freqset 
    
            
def get_rules(freqsets,mincof):
    rules=[]
    for freqset in freqsets.keys():
        if len(freqset)>1:
            rules_one_freqset=get_rules_from_one_freqset(freqsets,freqset,minconf)
            rules.extend(rules_one_freqset)
    return rules

    
def Apriori(data,minsum,minconf): 
    D=list(map(set,data)) #D记录事件列表的集合形式，便于查看是否子集、求支持度
    freqsets=get_freqsets(D,minsup)
    rules=get_rules(freqsets,minconf)
    return freqsets,rules


def load_data(filepath):
    data=[]
    with open(filepath,'r',encoding='utf-8-sig') as f:
        lines=f.readlines()
        for line in lines:
            data.append(line.strip().split(','))
    return data


if __name__=='__main__':
    time1=time.time()
    
    data=load_data('D:/学习/大三下/数据仓库与数据挖掘/guanlian_new.csv')
    minsup=0.3
    minconf=0.7
    freqsets,rules=Apriori(data,minsup,minconf)
    rules.sort(key=lambda x:x[2],reverse=True)
    print(len(freqsets))
    print(len(rules))
    
    time2=time.time()
    print(time2-time1)
    

#data=[[1,2,5],[2,4],[2,3],[1,2,4],[1,3],[2,3],[1,3],[1,2,3,5],[1,2,3]]
#data=[[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
            