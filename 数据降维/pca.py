import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def pca_me(X,nc):
    X_norm=X-X.mean(axis=0)
    conv=np.dot(X_norm.T,X_norm)
    eig_val,eig_vec = np.linalg.eig(conv) 
    sort_idx=eig_val.argsort()[::-1]
    req_idx=sort_idx[0:nc]
    req_eig_vec=eig_vec[:,req_idx]
    X_dec=np.dot(X_norm,req_eig_vec)
    return X_dec
    
    
if __name__ == '__main__':
    X=np.array([[6, 2,7,-3], [2, -1,6,-9], [-3, -2,4,-10], [-1, -1,-5,7.7], [2, 4,-1.68,-3.2], [3,-1, 9.9,-7]])
    X_dec_me=pca_me(X,nc=2)
    print(X_dec_me)
    plt.scatter(X_dec_me[:,0],X_dec_me[:,1])
    
    clf=PCA(n_components=2)
    X_dec=clf.fit_transform(X)
    print(X_dec)
    plt.scatter(X_dec[:,0],X_dec[:,1])
    