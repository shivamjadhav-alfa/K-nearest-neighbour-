import numpy as np
from collections import Counter

def euclidiandistance(x,y):
    return np.sqrt(np.sum((x-y)**2))

class KNN:
    def __init__(self,k=3):
        self.k=k
        
    def fit(self,X,Y):
        self.Xt=X
        self.yt=Y
        
    def pred(self,X):
        ypred=[self.predict(i) for i in X]
        return np.array(ypred)
    def predict(self,x):
        dist=[euclidiandistance(x,xt) for xt in self.Xt]
        k_n=np.argsort(dist)[:self.k]
        k_n_lab=[self.yt[i] for i in k_n]
        common=Counter(k_n_lab).most_common(1)
        return common[0][0]
        