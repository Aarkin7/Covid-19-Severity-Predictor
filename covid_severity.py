#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_excel("dataset.xlsx")
data


# ## Feature engineering

# In[4]:


data.columns = [x.lower().strip().replace(' ','_') for x in data.columns]

def miss_data(x):
    total = x.isnull().sum()
    percent = (x.isnull().sum()/x.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(x[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[5]:


for x in data.columns:
    if data[x].dtype=='float16' or  data[x].dtype=='float32' or  data[x].dtype=='float64':
        data[x].fillna(data[x].mean())

data = data.fillna(data.median())

for y in data.columns:
    if data[y].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(data[y].values))
        data[y] = lbl.transform(list(data[y].values))


# In[6]:


threshold = 0.92

corr_matrix = data.corr().abs()


upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


# In[7]:


to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %d columns to remove.' % (len(to_drop)))
dataset = data.drop(columns = to_drop)


# In[8]:


data_missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
data_missing.head()


# In[9]:


data_missing_ = data_missing.index[data_missing > 0.85]
all_missing = list(set(data_missing_))


dataset = dataset.drop(columns = all_missing)

dataset.info()


# In[10]:


cols = [x for x in dataset.columns if x not in ['patient_id','sars-cov-2_exam_result', 'patient_addmited_to_regular_ward_(1=yes,_0=no)', 'patient_addmited_to_semi-intensive_unit_(1=yes,_0=no)', 'patient_addmited_to_intensive_care_unit_(1=yes,_0=no)']]
new_df = dataset[cols]
new_df.head()


# In[11]:


X = new_df
y = dataset['sars-cov-2_exam_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=101)


# In[12]:


model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

feat_head = feat_importances.head(10)
feat_head.index


# ## Model to give covid result

# In[13]:


newdf = new_df[feat_head.index]

X = newdf
y = dataset['sars-cov-2_exam_result']
X


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)


# In[15]:


accuracy_lst =[]

def model_assess(model, name='Default'):
    model.fit(X_train, y_train)
    prds = model.predict(X_test)
    model_acc = accuracy_score(y_test, prds)
    accuracy_lst.append(100*model_acc)
    print('---', name, '---', '\n',
          confusion_matrix(y_test, prds), '\n',
          'Accuracy:', (accuracy_score(y_test, prds)), '\n',
          'Classification Report:', (classification_report(y_test, prds)))
    


# In[16]:


# Logistic Regression
lg = LogisticRegression()
model_assess(lg, 'Logistic Regression')
lg.fit(X_train,y_train)

# SVM
svm = SVC()
model_assess(svm, 'SVM')
svm.fit(X_train,y_train)


# In[17]:


cross_acc = []

ca_lg = cross_val_score(lg, X_train, y_train, scoring='accuracy')
ca_lg = ca_lg.mean()
cross_acc.append(100*ca_lg)



ca_svm = cross_val_score(svm, X_train, y_train, scoring='accuracy')
ca_svm = ca_svm.mean()
cross_acc.append(100*ca_svm)


# ## Admisiion to ward if patient covid positive: model of interest

# ### 0- no need to hospitialze, 1-general, 2 -icu

# In[18]:


covid_positive = dataset[dataset['sars-cov-2_exam_result'] == 1]

admission = []  

def multiclass_target(row):
    check = 0
    check += 1 if (row['patient_addmited_to_regular_ward_(1=yes,_0=no)'] == 1) else 0
    check += 2 if (row['patient_addmited_to_semi-intensive_unit_(1=yes,_0=no)'] == 1) else 0
    check += 2 if (row['patient_addmited_to_intensive_care_unit_(1=yes,_0=no)'] == 1) else 0
    row['target'] = check
    return row


# In[19]:


data_adm = covid_positive.apply(multiclass_target, axis=1)
data_adm.rename(columns = {'mean_corpuscular_hemoglobin_concentration': 'mchc','mean_corpuscular_hemoglobin_(mch)' : 'mch'},inplace= True)
data_adm

admit = data_adm[['patient_age_quantile','hematocrit','platelets',  'proteina_c_reativa_mg/dl', 'red_blood_cells', 'eosinophils','leukocytes', 'monocytes','influenza_b,_rapid_test'] ]
X=admit
y = data_adm['target']

admit


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[21]:


acc_lst = []

def model_assess(model, name='Default'):
    model.fit(X_train, y_train)
    prds = model.predict(X_test)
    model_acc = accuracy_score(y_test, prds)
    acc_lst.append(100*model_acc)
    print('---', name, '---', '\n',
          confusion_matrix(y_test, prds), '\n',
          'Accuracy:', (accuracy_score(y_test, prds)), '\n',
          'Classification Report:', (classification_report(y_test, prds)))


# In[22]:


lg = LogisticRegression()
model_assess(lg, 'Logistic Regression')
lg.fit(X_train, y_train)


# ### order of entering the values should be asper order of columns in admit table

# In[23]:


lg.predict([[16, 0.05, -0.12, -0.4, 0.0132, -0.32, -0.21,-0.1,0] ])


# In[24]:


lg.predict(X_test)

pickle.dump(lg,open('covid_model.pkl','wb'))
covid_model = pickle.load(open('covid_model.pkl','rb'))
