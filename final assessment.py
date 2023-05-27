#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sn


# In[120]:


traindata=pd.read_csv("C:/Users/Resh/Downloads/train_ctrUa4K.csv")
traindata.head()


# In[121]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[122]:


traindata['Gender']=le.fit_transform(traindata['Gender'])
traindata['Married']=le.fit_transform(traindata['Married'])
traindata['Self_Employed']=le.fit_transform(traindata['Self_Employed'])
traindata['Property_Area']=le.fit_transform(traindata['Property_Area'])
traindata['Loan_Status']=le.fit_transform(traindata['Loan_Status'])
traindata['Education']=le.fit_transform(traindata['Education'])

traindata.head()


# In[123]:


traindata.isna().sum()


# In[124]:


traindata1=traindata.replace(['3+'], '4')
traindata1
for i in ['LoanAmount','Dependents' ,'Loan_Amount_Term','Credit_History']:
    traindata1[i]=traindata1[i].fillna(traindata1[i].median())
traindata1


# In[125]:


traindata2=traindata1.drop(['Loan_ID'],axis=1)
traindata2


# In[126]:


traindata2.isna().sum()


# In[127]:


x=traindata2.drop('Loan_Status',axis=1)
y=traindata2['Loan_Status']


# In[128]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)



# In[129]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


# In[130]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()

model=lr.fit(x_train,y_train)
predictions=model.predict(x_test)
predictions


# In[131]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

print('Accuracy is',accuracy_score(y_test,predictions))


# In[132]:


from sklearn.neighbors import KNeighborsClassifier
metric_k =[]
neighbors=np.arange(3,15)

for k in neighbors:
    classifier=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
    classifier.fit(x_train,y_train)
    y_pred=classifier.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    metric_k.append(acc)


# In[133]:


classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print('Accuracy is',accuracy_score(y_test,y_pred))


# In[134]:


from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)
y_pred=dt_clf.predict(x_test)
print('accuracy is',accuracy_score(y_test,y_pred))


# In[135]:


from sklearn.ensemble import RandomForestClassifier 
rf_clf=RandomForestClassifier() 
rf_clf.fit(x_train,y_train) 
y_pred=rf_clf.predict(x_test)
print('accuracy is',accuracy_score(y_test,y_pred))


# In[142]:


y_pred.shape


# In[136]:


testdata=pd.read_csv("C:/Users/Resh/Downloads/test_lAUu6dG.csv")
testdata


# In[137]:


from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
testdata['Gender']=le1.fit_transform(testdata['Gender'])
testdata['Married']=le1.fit_transform(testdata['Married'])
testdata['Self_Employed']=le1.fit_transform(testdata['Self_Employed'])
testdata['Property_Area']=le1.fit_transform(testdata['Property_Area'])
testdata['Education']=le1.fit_transform(testdata['Education'])
testdata.head()


# In[138]:


testdata1=testdata.replace(['3+'], '4')
testdata1
for i in ['LoanAmount','Dependents' ,'Loan_Amount_Term','Credit_History']:
    testdata1[i]=testdata1[i].fillna(testdata1[i].median())
testdata1


# In[139]:


x_test1=testdata1.drop(['Loan_ID'],axis=1)
x_test1


# In[ ]:





# In[144]:


y_pred1=rf_clf.predict(x_test1)


# In[145]:


y_pred1=pd.DataFrame(y_pred1)
y_pred1


# In[146]:


testdata=pd.read_csv("C:/Users/Resh/Downloads/test_lAUu6dG.csv")
testdata1=testdata['Loan_ID']
testdata2=pd.DataFrame(testdata1)

testdata2


# In[147]:


testdata2['Loan_Status']=y_pred1


# In[149]:


testdata2 = testdata2.reset_index(drop=True)
testdata2


# In[155]:


testdata2['Loan_Status'].replace ({1: 'Y', 0: 'N'})
testdata3=pd.DataFrame(testdata2)
testdata3


# In[152]:


testdata1.to_csv('C:/Users/Resh/Downloads/assesment.csv')



# In[ ]:





# In[ ]:




