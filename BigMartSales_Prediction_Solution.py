#!/usr/bin/env python
# coding: utf-8

# ## This approach predicts sales 

# In[2]:


import pandas as pd
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_squared_error,roc_auc_score

from sklearn.linear_model import Lasso,Ridge,ElasticNet

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# In[3]:


train=pd.read_csv('train_v9rqX0R.csv')
test=pd.read_csv('test_AbJTz2l.csv')


# ## Exploratory Data Analysis

# ### Data Audit

# In[4]:


# User defined function to calculate descriptives of continuous variable
def udf_cont_var(x):
    return pd.Series([x.dtype,x.count(),x.isnull().sum(),x.nunique(),x.mean(),x.std(),x.var(),x.min(),x.quantile(0.01),
                      x.quantile(0.05),x.quantile(0.1),x.quantile(0.25),x.quantile(0.5),x.quantile(0.75),x.quantile(0.9),
                      x.quantile(0.95),x.quantile(0.99),x.max()], index=['DataType','N','NMISS','C','MEAN','STD','VAR','MIN',
                                                                        'P1','P5','P10','P25','P50','P75','P90',
                                                                        'P95','P99','MAX'])


# In[5]:


# User defined function to calculate descriptives of categorical variable
def udf_cat_var(x):
    return pd.Series([x.dtype,x.nunique(),x.count(),x.isnull().sum(),
                     x.value_counts().sort_values(ascending=False).index[0],
                     x.value_counts().sort_values(ascending=False).values[0],
                     (x.value_counts().sort_values(ascending=False).values[0]/x.shape[0])*100],
                    index=['DataType','C','N','NMISS','MODE','FREQ','PERCENT'])


# In[6]:


# Imputing item outlet sales value in test data as null as these needs to be predicted 
test['Item_Outlet_Sales']=np.nan


# In[7]:


print(train['Item_Identifier'].nunique(),train['Outlet_Identifier'].nunique())
print(test['Item_Identifier'].nunique(),test['Outlet_Identifier'].nunique())


# In[8]:


print(len(set(test['Item_Identifier'].unique()).intersection(train['Item_Identifier'].unique())),test['Item_Identifier'].nunique())
print(len(set(test['Outlet_Identifier'].unique()).intersection(train['Outlet_Identifier'].unique())),test['Outlet_Identifier'].nunique())


# In[9]:


# Consider only records in training data which contains outlets specified in test records
train=train[train['Item_Identifier'].apply(lambda x:x in test['Item_Identifier'].unique())]


# In[10]:


continuous_var=['Item_Weight','Item_Visibility','Item_MRP','Item_Outlet_Sales']
categorical_var=['Item_Identifier','Outlet_Identifier','Item_Fat_Content','Outlet_Establishment_Year','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']


# In[11]:


# Evaluting descriptives of continuous variables in training data - item_weight contains missing values for 1440 records
# Item_Outlet_Sales appears to have outliers
train[continuous_var].apply(lambda x:udf_cont_var(x))


# In[12]:


# Evaluting descriptives of continuous variables in test data - item_weight contains missing values for 976 records
test[continuous_var].apply(lambda x:udf_cont_var(x))


# In[13]:


# Evaluting descriptives of categorical variables in training data - outlet_size contains missing values for 2373 records. 
train[categorical_var].apply(lambda x:udf_cat_var(x))


# In[14]:


# Evaluting descriptives of categorical variables in test data - outlet_size contains missing values for 1606 records. 
test[categorical_var].apply(lambda x:udf_cat_var(x))


# In[15]:


# Capping outliers for Item_Outlet_Sales in train data - Capping at 1% (minimum) and 99% (maximum) quantile values
train['Item_Outlet_Sales']=train['Item_Outlet_Sales'].clip(lower=train['Item_Outlet_Sales'].quantile(0.01),upper=train['Item_Outlet_Sales'].quantile(0.99))


# In[16]:


train['Outlet_Size']=train['Outlet_Size'].fillna('Not Available')
test['Outlet_Size']=test['Outlet_Size'].fillna('Not Available')


# ### Data Transformation

# In[18]:


train['IsTest']=0
test['IsTest']=1
all_data=pd.concat([train,test],axis=0)


# In[19]:


init_var_list=['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier','Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'Item_Outlet_Sales']


# ### Feature Engineering - Adverserial Validation

# In[20]:


x_cols=list(set(init_var_list).difference(['Item_Outlet_Sales']))
X=all_data[init_var_list]
Y=all_data['IsTest']

train_X_adv,test_X_adv,train_Y_adv,test_Y_adv=train_test_split(X[x_cols],Y,test_size=0.3,random_state=38)
adv_mod=CatBoostClassifier(random_state=42,verbose=0)
categorical_features=list(set(train.select_dtypes(['object']).columns).intersection(x_cols))

adv_mod.fit(train_X_adv,train_Y_adv,eval_set=(test_X_adv,test_Y_adv),cat_features=categorical_features,early_stopping_rounds=1)

print(roc_auc_score(test_Y_adv,adv_mod.predict(test_X_adv)))

adv_fe=pd.concat([pd.Series(adv_mod.feature_names_),pd.Series(adv_mod.feature_importances_)],axis=1)
adv_fe.columns=['Feature','Importance']
print(adv_fe.sort_values('Importance',ascending=False))

intermediary_var_list=list(adv_fe['Feature'])


# In[21]:


all_data=pd.concat([X.reset_index(),Y.reset_index()],axis=1)
X_train=all_data.loc[all_data['IsTest']==0,intermediary_var_list]
X_test=all_data.loc[all_data['IsTest']==1,intermediary_var_list]


# ## Feature Engineering

# In[22]:


# CatBoost
cb_mod_fe=CatBoostRegressor(random_state=42,verbose=0)
cb_mod_fe.fit(X_train,train['Item_Outlet_Sales'],cat_features=categorical_features)
print(np.sqrt(mean_squared_error(train['Item_Outlet_Sales'],cb_mod_fe.predict(X_train))))
cb_mod_fe_df=pd.concat([pd.Series(cb_mod_fe.feature_names_),pd.Series(cb_mod_fe.feature_importances_)],axis=1)
cb_mod_fe_df.columns=['Feature','Importance']
print(cb_mod_fe_df.sort_values('Importance',ascending=False))
cb_feature_list=list(cb_mod_fe_df.loc[cb_mod_fe_df['Importance']>4.1,'Feature'])
final_features=cb_feature_list.copy()


# ## Train Test Split

# In[36]:


train_X,test_X,train_Y,test_Y=train_test_split(X_train[final_features],train['Item_Outlet_Sales'],test_size=0.3,random_state=42)
prediction_X=X_test[final_features]


# ## Model Building

# In[37]:


# CatBoost Regressor
cb_mod=CatBoostRegressor(random_state=42,verbose=0)
categorical_features=list(train_X.select_dtypes(['object']).columns)
cb_mod.fit(train_X,train_Y,eval_set=(test_X,test_Y),cat_features=categorical_features,early_stopping_rounds=1)
print(np.sqrt(mean_squared_error(train_Y,cb_mod.predict(train_X))))
print(np.sqrt(mean_squared_error(test_Y,cb_mod.predict(test_X))))
prediction_file=test.loc[:,['Item_Identifier','Outlet_Identifier']]
prediction_file['Item_Outlet_Sales']=cb_mod.predict(prediction_X)
prediction_file.to_csv('solution.csv',index=False)

