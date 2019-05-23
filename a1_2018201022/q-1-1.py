#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pprint
from sklearn.model_selection import train_test_split
eps = np.finfo(float).eps
from numpy import log2 as log
df =pd.read_csv('/home/neelesh/Downloads/train.csv')
df = df[['Work_accident','sales','promotion_last_5years','salary','left']]
mylist= ['Work_accident','sales','promotion_last_5years','salary','left']
X = df[['Work_accident','sales','promotion_last_5years','salary']] 
Y = df[['left']] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

df=pd.concat([X_train,Y_train], axis=1)


# In[3]:


def impurity(x):
    return -x*np.log2(x+eps)
def calc_entropy(df):
    entropy_node = 0 
    values= df.left.unique() 
    for value in values:
        x = df.left.value_counts()[value]*1.0/len(df.left) 
        entropy_node += impurity(x)
    return entropy_node

def obtain_subtable_length(df,key,variable,target=None):
    if(target is not None):
        return len(df[key][df[key]==variable][df.left ==target])
    else:
        return len(df[key][df[key]==variable]) 
    
def entropy_sub(df,variable,key):
        target_variables = df.left.unique() 
        entropy_each_feature = 0
        for target in target_variables:
                x=0
                n=obtain_subtable_length(df,key,variable,target) 
                d=obtain_subtable_length(df,key,variable)
                if(n!=0):
                     x = n/(d+eps) 
                entropy_each_feature += impurity(x)
        return entropy_each_feature
    
def calc_entropy_attribute(df,attribute):
    
    variables = df[attribute].unique() 
    entropy_attribute = 0
    for variable in variables:
        entropy_res=entropy_sub(df,variable,attribute)
        if(len(df)!=0):                                          
             y= len(df[attribute][df[attribute]==variable])*1.0/len(df)
        if(y!=0):                                          
             entropy_attribute +=y*entropy_res
    return abs(entropy_attribute)

def find_best_attribute(df,mylist):
    maxgain=None
    ans=None
    E=calc_entropy(df)
    for key in df.keys()[:-1]:
        if(key in mylist):
                I=calc_entropy_attribute(df,key)
                Gain=(E-I)
                if(maxgain is None or Gain>maxgain):
                                 maxgain=Gain
                                 ans=key
    return ans

def buildTree(df,mylist,tree=None): 
    node = find_best_attribute(df,mylist)
    attValue = np.unique(df[node])
    prev=None
    x=None
    flag=0
    if tree is None:                    
        tree={}
        tree[node] = {}

    for value in attValue:
        newtable = df[df[node] == value].reset_index(drop=True)
        clValue,counts = np.unique(newtable['left'],return_counts=True)                     

        if len(counts)==1:
            tree[node][value] = clValue[0]
            x=clValue[0]
            if(prev is None):
                    prev=x
            elif(x!=prev or x==-1):
                    flag=1
        else:
            size=len(newtable)
            zero=newtable['left'].value_counts()[0]
            one=newtable['left'].value_counts()[1]
            checkarr=list(newtable)[0:-1]
            i=0
            while(i+1<size):
                f=0
                for col in checkarr:
                    if(newtable[col][i]!=newtable[col][i+1]):
                             f=1
                             break
                i+=1 
                if(f==1):
                    break
            if(i+1<size):
                temp=mylist[:]
                mylist.remove(node)
                x,tree[node][value] = buildTree(newtable,mylist[:])
                if(tree[node][value] is None):
                    tree[node][value]=x
                if(prev is None):
                    prev=tree[node][value]
                elif(x!=prev or x==-1):
                    x=-1
                    flag=1
                mylist=temp[:]
            else:
                if(zero>=one):
                    tree[node][value] = 0
                elif(one>zero):
                    tree[node][value] = 1
                x=tree[node][value] 
                if(prev is None):
                    prev=x
                elif(x!=prev or x==-1):
                    flag=1    
                                   
    if(x is None or flag==1):
        return -1,tree
    else:
        return x,None
def predict(tree,inst):
        for nodes in tree.keys():        

            try:
                value = inst[nodes]

                tree = tree[nodes][value]

            except:
                return 0
            prediction = 0

            if type(tree) is dict:
                prediction = predict(tree,inst)
            else:
                prediction = tree
                break;                            

        return prediction
def calculate_recall_precision(original,res):
        TP=0
        FP=0
        TN= 0
        FN= 0
        f1_score=0
        for i in range(0, len(original)):

                if res[i] == 1:
                    if res[i] == original[i]:
                        TP+= 1
                    else:
                        FP+= 1
                else:
                    if res[i] == original[i]:
                        TN+= 1
                    else:
                        FN+= 1

        precision=0
        recall=0
        if(TP!=0 or TN!=0):
                accuracy = (TP+TN)*1.0/(TP + TN +FP +FN)
        if(TP!=0):
                precision = TP*1.0/(TP + FP)
                recall = TP*1.0/(TP + FN)
                f1_score = 2 / ((1 / precision) + (1 / recall))
        print "True +ve=",TP,"True -ve=",TN,"False +ve=",FP,"False -ve=",FN                    

        return accuracy*100, precision*100, recall*100,f1_score

x,t=buildTree(df,mylist[:])
pprint.pprint(t)
print "------------------------------------------------------------------------"
print "-----------------------RESULTS------------------------------------------"
print "------------------------------------------------------------------------"
res=[]
for i in range(0,len(X_test)):
        test=X_test.iloc[i]
        p=predict(t,test)
        res.append(p) 
Y_test=np.array(Y_test['left'])
accuracy, precision, recall,f1_score = calculate_recall_precision(Y_test,res)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)

