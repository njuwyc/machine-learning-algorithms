import time

class treenode:
    def __init__(self,name,numOccur,parentNode):
        self.name=name
        self.count=numOccur
        self.parent=parentNode
        self.nodelink=None
        self.children={}
    def inc(self,numOccur):
        self.count=self.count+numOccur
    def display(self,ind=1):
        print('   '*ind,self.name,self.count)
        for child in self.children.values():
            child.display(ind+1)

        
def insert_tran(local_tran,rootnode,headertable):
    if local_tran==[]:
        pass
    elif local_tran[0] not in rootnode.children.keys():
        newnode=treenode(local_tran[0],1,rootnode)
        rootnode.children[local_tran[0]]=newnode
        #print(rootnode.name,'的子节点为',rootnode.children)
        if headertable[local_tran[0]].nodelink==None:
            headertable[local_tran[0]].nodelink=newnode
        else:
            tailnode=headertable[local_tran[0]]
            while tailnode.nodelink!=None:
                tailnode=tailnode.nodelink
            tailnode.nodelink=newnode
        insert_tran(local_tran[1:],rootnode.children[local_tran[0]],headertable)
    else:
        #print(rootnode.name,'的子节点为',rootnode.children)
        rootnode.children[local_tran[0]].count+=1
        insert_tran(local_tran[1:],rootnode.children[local_tran[0]],headertable)
    
    
def create_fptree(data,minsup2):
    single_freqset={}
    for tran in data:
        for item in tran:
            single_freqset[item]=single_freqset.get(item,0)+1
    
    for item in single_freqset.copy().keys():
        if single_freqset[item]<minsup2:
            del single_freqset[item]
            
    headertable={item:treenode(item,1,None) for item in single_freqset.keys()}
    rootnode=treenode('empty',1,None)
    for tran in data:
        local_tran=[]
        for item in tran:
            if item in single_freqset.keys():
                local_tran.append(item)
        local_tran.sort(key=lambda x:(single_freqset[x],x),reverse=True)
        #print(local_tran)
        if len(local_tran)>0:
            insert_tran(local_tran,rootnode,headertable)
    headertable=sorted(headertable.items(),key=lambda x:single_freqset[x[0]],reverse=True)
    return rootnode,headertable


def get_prepath(leafnode,fptree):
    prepath=[]
    while leafnode.parent.parent!=None:
        prepath.append(leafnode.parent.name)
        leafnode=leafnode.parent
    prepath=prepath[::-1]
    return prepath
    
    
def get_cpb(prepaths_onenode):
    cpb=[]
    for prepath in prepaths_onenode:
        for i in range(prepath[1]):
            cpb.append(prepath[0])
    return cpb
        
        
def mine_fptree(fptree,headertable,minsup2,prefix,freqsets):
    for header in headertable[::-1]:
        #print(header)
        currentnode=header[1]
        #print(prefix)
        newfreqset=prefix.copy()
        newfreqset.add(header[0])
        #print(newfreqset)
        freqsets.append(newfreqset)
        #print('freqsets:',freqsets)
        prepaths_onenode=[]
        while currentnode.nodelink!=None:
            leafnode=currentnode.nodelink
            prepath=get_prepath(leafnode,fptree)
            prepaths_onenode.append((prepath,leafnode.count))
            currentnode=currentnode.nodelink
        #print(prepaths_onenode)
        cpb=get_cpb(prepaths_onenode)
        #print(cpb)
        cpbfptree,cpbheadertable=create_fptree(cpb,minsup2)
        #cpbfptree.display()
        if cpbheadertable!=None:
            mine_fptree(cpbfptree,cpbheadertable,minsup2,newfreqset,freqsets)
            

def FP_growth(data,minsup):
    minsup2=len(data)*minsup
    fptree,headertable=create_fptree(data,minsup2)
    '''
    for header in headertable:
        tailnode=header[1]
        while tailnode.nodelink!=None:
            print(tailnode.nodelink.name)
            tailnode=tailnode.nodelink  
    '''
    freqsets=[]
    mine_fptree(fptree,headertable,minsup2,set([]),freqsets)
    return freqsets
    

if __name__=='__main__':
    data=[['f','a','c','d','g','i','m','p'],['a','b','c','f','l','m','o'],
      ['b','f','h','j','o','w'],['b','c','k','s','p'],
      ['a','f','c','e','l','p','m','n']]  
    minsup=3/6
    freqsets=FP_growth(data,minsup)
    print(freqsets)
    
    
    '''
    data=[['r','z','h','j','p'],
                  ['z','y','x','w','v','u','t','s'],
                  ['z'],
                  ['r','x','n','o','s'],
                  ['y','r','x','z','q','t','p'],
                  ['y','z','x','e','q','s','t','m']]
    '''
        
      

    

        
