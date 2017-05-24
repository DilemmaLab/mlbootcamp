
# coding: utf-8

# In[17]:

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
get_ipython().magic('matplotlib inline')
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


# In[18]:

x_test = pd.read_csv('x_test.csv',sep=';',header=None)
x_train = pd.read_csv('x_train.csv',sep=';',header=None)
y = pd.read_csv('y_train.csv',sep=';',header=None)
total_X = pd.concat((x_train, x_test))


# In[19]:

n_rows = x_train.shape[0]
x_train_test = np.array(total_X)

x_train = x_train_test[:n_rows, :]
x_test = x_train_test[n_rows:, :]
print('Shape train', x_train.shape)
print('Shape test', x_test.shape)


# In[20]:

for i in xrange(223):
    print np.mean(x_train_test[:,i])


# In[21]:

average_list = np.zeros((223,1))
for i in xrange(223):
    average_list[i] = np.mean(x_train_test[:,i])


# In[22]:

too_less = []
too_much = []
for i in xrange(223):
    if average_list[i]>0.9:
        print 'larger  0.9   ',i
        too_much.append(i)
    if average_list[i]<0.05:
        print 'smaller 0.05   ',i
        too_less.append(i)


# In[23]:

x_train_test = x_train_test[:,malo]
x_train = x_train_test[:n_rows, :]
x_test = x_train_test[n_rows:, :]


# ### Probability
cycle = 100
c = 0
y = pd.read_csv('y_train.csv',sep=';',header=None)
for i in xrange(cycle):
    rfc = RandomForestClassifier(n_estimators=np.random.randint(200,800))
    rfc.fit(X=x_train,y=y)
    pred = rfc.predict_proba(x_test)
    df = pd.DataFrame(pred)
    df.to_csv(str(i)+'test.csv',index=None,header = None)
    c+=1
    if c % 50 ==0:
        print cdf1 = pd.read_csv('0test.csv',header=None)
for i in xrange(1,cycle):
    df1 = df1 + pd.read_csv(str(i)+'test.csv',header=None)for i in xrange(2327):
    for j in xrange(5):
        if df1.iloc[i][j] == np.max(df1.iloc[i]):
            df1.iloc[i][j] = 9999
        else:
            df1.iloc[i][j] = 0
for i in xrange(2327):
    for j in xrange(5):
        if df1.iloc[i][j] ==9999:
            df1.iloc[i][j] = j
        else:
            df1.iloc[i][j] = 0
df1['y'] = 0
for i in xrange(2327):
    df1.y.values[i] = np.sum(df1.iloc[i][0:5])x = df1['y']
x.to_csv('proba_rf.csv',header=None,index=None)
# ### Classes

# In[24]:

cycle = 20
c = 0
y = pd.read_csv('y_train.csv',sep=';',header=None)
for i in xrange(cycle):
    rfc = RandomForestClassifier(n_estimators=np.random.randint(200,800))
    rfc.fit(X=x_train,y=y)
    pred = rfc.predict(x_test)
    df = pd.DataFrame(pred)
    df.to_csv(str(i)+'test.csv',index=None,header = None)
    c+=1
    if c % 10 ==0:
        print c


# In[25]:

df1 = pd.read_csv('0test.csv',header=None)
for i in xrange(1,cycle):
    df1 = pd.concat((df1,pd.read_csv(str(i)+'test.csv',header=None)),axis = 1)
from scipy.stats import mode

merg = df1
predict = np.zeros((2327))
predict_mean = np.zeros((2327))

for i in xrange(len(merg)):
    predict[i] = mode(merg.ix[i,:]).mode
    predict_mean[i] = round(np.mean(merg.ix[i,:]))
    
predict = pd.DataFrame(predict)
predict.to_csv('final'+str(cycle)+'.csv',index=None,header = None)

