#!/usr/bin/env python
# coding: utf-8

# ## This approach predicts sales 

# In[1]:


version_number=2


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


train=pd.read_csv('C:/Users/sidha/OneDrive/Desktop/Work/Hckthon/ABB/Data/train_v9rqX0R.csv')
test=pd.read_csv('C:/Users/sidha/OneDrive/Desktop/Work/Hckthon/ABB/Data/test_AbJTz2l.csv')


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


# User defined function to ceate dummy variable
def udf_dummy_var(input_df,column_name):
    return pd.concat([input_df,pd.get_dummies(input_df[column_name],prefix=column_name,drop_first='true')],axis=1).drop(column_name,axis=1)


# In[7]:


# Creating derived variable 'Outlet Age' by subtrating the outlet establishment year from current year
current_year=datetime.now().year
train['Outlet_Age']=train['Outlet_Establishment_Year'].apply(lambda x:current_year-x)
test['Outlet_Age']=test['Outlet_Establishment_Year'].apply(lambda x:current_year-x)

# Imputing item outlet sales value in test data as null as these needs to be predicted 
test['Item_Outlet_Sales']=np.nan


# In[8]:


print(train['Item_Identifier'].nunique(),train['Outlet_Identifier'].nunique())
print(test['Item_Identifier'].nunique(),test['Outlet_Identifier'].nunique())


# In[9]:


print(len(set(test['Item_Identifier'].unique()).intersection(train['Item_Identifier'].unique())),test['Item_Identifier'].nunique())
print(len(set(test['Outlet_Identifier'].unique()).intersection(train['Outlet_Identifier'].unique())),test['Outlet_Identifier'].nunique())


# In[10]:


# Consider only records in training data which contains outlets specified in test records
train=train[train['Item_Identifier'].apply(lambda x:x in test['Item_Identifier'].unique())]


# In[11]:


continuous_var=['Item_Weight','Item_Visibility','Item_MRP','Outlet_Age','Item_Outlet_Sales']
categorical_var=['Item_Identifier','Outlet_Identifier','Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']


# In[12]:


# Evaluting descriptives of continuous variables in training data - item_weight contains missing values for 1440 records
# Item_Outlet_Sales appears to have outliers
train[continuous_var].apply(lambda x:udf_cont_var(x))


# In[13]:


# Evaluting descriptives of continuous variables in test data - item_weight contains missing values for 976 records
test[continuous_var].apply(lambda x:udf_cont_var(x))


# In[14]:


# Evaluting descriptives of categorical variables in training data - outlet_size contains missing values for 2373 records. 
# Item_Identifier, Outlet_Identifier, Item_Type have high cardinality and cause high dimensionality if one-got encoding followed
train[categorical_var].apply(lambda x:udf_cat_var(x))


# In[15]:


# Evaluting descriptives of categorical variables in test data - outlet_size contains missing values for 1606 records. 
# Item_Identifier, Outlet_Identifier, Item_Type have high cardinality and cause high dimensionality if one-got encoding followed
test[categorical_var].apply(lambda x:udf_cat_var(x))


# In[16]:


# Capping outliers for Item_Outlet_Sales in train data - Capping at 1% (minimum) and 99% (maximum) quantile values
train['Item_Outlet_Sales']=train['Item_Outlet_Sales'].clip(lower=train['Item_Outlet_Sales'].quantile(0.01),upper=train['Item_Outlet_Sales'].quantile(0.99))


# In[17]:


train['Outlet_Size_Mapped']=train['Outlet_Size'].map({'Small':1,'Medium':2,'High':3})
test['Outlet_Size_Mapped']=test['Outlet_Size'].map({'Small':1,'Medium':2,'High':3})


# ### Data Transformation

# In[18]:


for dummy_col in ['Item_Fat_Content','Outlet_Location_Type','Outlet_Type']:
    train[dummy_col]=train[dummy_col].astype('category')
    train=udf_dummy_var(train,dummy_col)
    
    test[dummy_col]=test[dummy_col].astype('category')
    test=udf_dummy_var(test,dummy_col)    


# In[19]:


train.columns


# In[20]:


# Target encoding of Item_Identifier, Outlet_Identifier and Item_Type using measure of central tendency (mean) and dispersion (min, max, variance and standard deviation)
encoded_columns=[]
aggregation_df=train.groupby('Item_Identifier')['Item_Outlet_Sales'].agg(['min','mean','max','var','std']).add_suffix('_Item_Identifier_Sales')
encoded_columns.extend(aggregation_df.columns)

print(train.shape[0],test.shape[0])
train=pd.merge(train,aggregation_df,on='Item_Identifier',how='inner')
test=pd.merge(test,aggregation_df,on='Item_Identifier',how='inner')
print(train.shape[0],test.shape[0])

aggregation_df=train.groupby('Outlet_Identifier')['Item_Outlet_Sales'].agg(['min','mean','max','var','std']).add_suffix('_Outlet_Identifier_Sales')
encoded_columns.extend(aggregation_df.columns)

print(train.shape[0],test.shape[0])
train=pd.merge(train,aggregation_df,on='Outlet_Identifier',how='inner')
test=pd.merge(test,aggregation_df,on='Outlet_Identifier',how='inner')
print(train.shape[0],test.shape[0])

aggregation_df=train.groupby('Item_Type')['Item_Outlet_Sales'].agg(['min','mean','max','var','std']).add_suffix('_Item_Type_Sales')
encoded_columns.extend(aggregation_df.columns)

print(train.shape[0],test.shape[0])
train=pd.merge(train,aggregation_df,on='Item_Type',how='inner')
test=pd.merge(test,aggregation_df,on='Item_Type',how='inner')
print(train.shape[0],test.shape[0])


# In[21]:


print(encoded_columns)
# print(train['Item_Outlet_Sales'].corr(train['Item_MRP']))


# In[22]:


train['IsTest']=0
test['IsTest']=1
all_data=pd.concat([train,test],axis=0)


# In[23]:


init_var_list=['Item_Weight','Item_Visibility','Item_MRP','Outlet_Age','Item_Fat_Content_Low Fat', 'Item_Fat_Content_Regular','Item_Fat_Content_low fat', 'Item_Fat_Content_reg', 'Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3','Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2','Outlet_Type_Supermarket Type3','min_Item_Identifier_Sales', 'mean_Item_Identifier_Sales', 'max_Item_Identifier_Sales', 'var_Item_Identifier_Sales', 'std_Item_Identifier_Sales', 'min_Outlet_Identifier_Sales', 'mean_Outlet_Identifier_Sales', 'max_Outlet_Identifier_Sales', 'var_Outlet_Identifier_Sales', 'std_Outlet_Identifier_Sales', 'min_Item_Type_Sales', 'mean_Item_Type_Sales', 'max_Item_Type_Sales', 'var_Item_Type_Sales', 'std_Item_Type_Sales','Outlet_Size_Mapped']


# ### Missing Value Imputation

# In[24]:


# Missing value imputation using adverserial validation
missing_cols=['Outlet_Size_Mapped','Item_Weight','var_Item_Identifier_Sales','std_Item_Identifier_Sales']
X=all_data[init_var_list]
Y=all_data['IsTest']

imputer_model=KNNImputer(n_neighbors=2, weights='uniform')
X_imputed = pd.DataFrame(imputer_model.fit_transform(X), columns=X.columns)
x_cols=list(set(init_var_list).difference(['min_Item_Identifier_Sales','mean_Item_Identifier_Sales','max_Item_Identifier_Sales']))

train_X_adv,test_X_adv,train_Y_adv,test_Y_adv=train_test_split(X_imputed[x_cols],Y,test_size=0.3,random_state=38)
adv_mod=CatBoostClassifier(random_state=42,verbose=0)
adv_mod.fit(train_X_adv,train_Y_adv,eval_set=(test_X_adv,test_Y_adv),early_stopping_rounds=1)
print(roc_auc_score(test_Y_adv,adv_mod.predict(test_X_adv)))

adv_fe=pd.concat([pd.Series(adv_mod.feature_names_),pd.Series(adv_mod.feature_importances_)],axis=1)
adv_fe.columns=['Feature','Importance']
print(adv_fe.sort_values('Importance',ascending=False))

intermediary_var_list=list(adv_fe['Feature'])


# In[25]:


intermediary_var_list


# In[26]:


all_data=pd.concat([X_imputed.reset_index(),Y.reset_index()],axis=1)
X_train=all_data.loc[all_data['IsTest']==0,intermediary_var_list]
X_test=all_data.loc[all_data['IsTest']==1,intermediary_var_list]


# ## Feature Engineering

# In[27]:


# Random Forest
rf_mod_fe=RandomForestRegressor(random_state=38,verbose=0)
rf_mod_fe.fit(X_train,train['Item_Outlet_Sales'])
print(np.sqrt(mean_squared_error(train['Item_Outlet_Sales'],rf_mod_fe.predict(X_train))))
rf_mod_fe_df=pd.concat([pd.Series(rf_mod_fe.feature_names_in_),pd.Series(rf_mod_fe.feature_importances_)],axis=1)
rf_mod_fe_df.columns=['Feature','Importance']
print(rf_mod_fe_df.sort_values('Importance',ascending=False))
rf_feature_list=list(rf_mod_fe_df.loc[rf_mod_fe_df['Importance']>0.04,'Feature'])


# In[28]:


# XGBoost
xgb_mod_fe=XGBRegressor(random_state=38)
xgb_mod_fe.fit(X_train,train['Item_Outlet_Sales'])
print(np.sqrt(mean_squared_error(train['Item_Outlet_Sales'],xgb_mod_fe.predict(X_train))))
xgb_mod_fe_df=pd.concat([pd.Series(X_train.columns),pd.Series(xgb_mod_fe.feature_importances_)],axis=1)
xgb_mod_fe_df.columns=['Feature','Importance']
print(xgb_mod_fe_df.sort_values('Importance',ascending=False))
xgb_feature_list=list(xgb_mod_fe_df.loc[xgb_mod_fe_df['Importance']>0.03,'Feature'])


# In[29]:


# LightGBM
lgbm_mod_fe=LGBMRegressor(random_state=38)
lgbm_mod_fe.fit(X_train,train['Item_Outlet_Sales'])
print(np.sqrt(mean_squared_error(train['Item_Outlet_Sales'],lgbm_mod_fe.predict(X_train))))
lgbm_mod_fe_df=pd.concat([pd.Series(X_train.columns),pd.Series(lgbm_mod_fe.feature_importances_)],axis=1)
lgbm_mod_fe_df.columns=['Feature','Importance']
print(lgbm_mod_fe_df.sort_values('Importance',ascending=False))
lgbm_feature_list=list(lgbm_mod_fe_df.loc[lgbm_mod_fe_df['Importance']>100,'Feature'])


# In[30]:


# CatBoost
cb_mod_fe=CatBoostRegressor(random_state=42,verbose=0)
cb_mod_fe.fit(X_train,train['Item_Outlet_Sales'])
print(np.sqrt(mean_squared_error(train['Item_Outlet_Sales'],cb_mod_fe.predict(X_train))))
cb_mod_fe_df=pd.concat([pd.Series(cb_mod_fe.feature_names_),pd.Series(cb_mod_fe.feature_importances_)],axis=1)
cb_mod_fe_df.columns=['Feature','Importance']
print(cb_mod_fe_df.sort_values('Importance',ascending=False))
cb_feature_list=list(cb_mod_fe_df.loc[cb_mod_fe_df['Importance']>5,'Feature'])


# In[31]:


# Ridge Regularisation
param_grid=[{'alpha': 10.0 ** np.arange(-5, 6)}]
ridge_fe_cv = GridSearchCV(Ridge(), param_grid=param_grid, cv=5,n_jobs=-1,verbose=0)
ridge_fe_cv.fit(X_train,train['Item_Outlet_Sales'])
ridge_fe=ridge_fe_cv.best_estimator_
ridge_fe_df=pd.concat([pd.Series(ridge_fe.feature_names_in_),pd.Series(ridge_fe.coef_)],axis=1)
ridge_fe_df.columns=['Feature','Importance']
ridge_fe_df['Abs_Importance']=ridge_fe_df['Importance'].apply(lambda x:np.abs(x))
(ridge_fe_df.sort_values('Abs_Importance',ascending=False)).drop('Abs_Importance',axis=1)
ridge_feature_list=list(ridge_fe_df.loc[ridge_fe_df['Abs_Importance']>0.1,'Feature'])


# In[32]:


# Lasso Regularisation
param_grid=[{'alpha': 10.0 ** np.arange(-5, 6),'fit_intercept': [True, False]}]
lasso_fe_cv = GridSearchCV(Lasso(), param_grid=param_grid, cv=5,n_jobs=-1, verbose=0)
lasso_fe_cv.fit(X_train,train['Item_Outlet_Sales'])
lasso_fe=lasso_fe_cv.best_estimator_
lasso_fe_df=pd.concat([pd.Series(lasso_fe.feature_names_in_),pd.Series(lasso_fe.coef_)],axis=1)
lasso_fe_df.columns=['Feature','Importance']
lasso_fe_df['Abs_Importance']=lasso_fe_df['Importance'].apply(lambda x:np.abs(x))
(lasso_fe_df.sort_values('Abs_Importance',ascending=False)).drop('Abs_Importance',axis=1)
lasso_feature_list=list(lasso_fe_df.loc[lasso_fe_df['Abs_Importance']>0.1,'Feature'])


# In[33]:


# ElasticNet Regularisation
param_grid=[{'alpha': 10.0 ** np.arange(-5, 6),"l1_ratio": np.arange(0.0, 1.0, 0.1)}]
elastic_net_fe_cv = GridSearchCV(ElasticNet(), param_grid=param_grid, cv=5,n_jobs=-1, verbose=0)
elastic_net_fe_cv.fit(X_train,train['Item_Outlet_Sales'])
elastic_net_fe=elastic_net_fe_cv.best_estimator_
elastic_net_fe_df=pd.concat([pd.Series(elastic_net_fe.feature_names_in_),pd.Series(elastic_net_fe.coef_)],axis=1)
elastic_net_fe_df.columns=['Feature','Importance']
elastic_net_fe_df['Abs_Importance']=elastic_net_fe_df['Importance'].apply(lambda x:np.abs(x))
(elastic_net_fe_df.sort_values('Abs_Importance',ascending=False)).drop('Abs_Importance',axis=1)
elastic_net_feature_list=list(elastic_net_fe_df.loc[elastic_net_fe_df['Abs_Importance']>1,'Feature'])


# ## Train Test Split and Model Training

# In[34]:


final_features_dict={'rf':rf_feature_list,'xgb':xgb_feature_list,'lgbm':lgbm_feature_list,'cb':cb_feature_list,'ridge':ridge_feature_list,
'lasso':lasso_feature_list,'elastic_net':elastic_net_feature_list}
for feature_engg_method in list(final_features_dict.keys()):
    final_features=final_features_dict[feature_engg_method]    
    
    # Train Test Split
    train_X,test_X,train_Y,test_Y=train_test_split(X_train[final_features],train['Item_Outlet_Sales'],test_size=0.3,random_state=42)

    # Standardising the data for faster covergence during model training
    sc=StandardScaler()
    train_X_sc=pd.DataFrame(sc.fit_transform(train_X),columns=train_X.columns)
    test_X_sc=pd.DataFrame(sc.fit_transform(test_X),columns=test_X.columns)

    prediction_X_sc=pd.DataFrame(sc.fit_transform(X_test[final_features]),columns=final_features)
    
    # Random Forest
    param_grid=[{'n_estimators': [100, 150, 200, 250, 300], 'max_depth':[5,10,15,20]}]
    rf_mod_cv=GridSearchCV(RandomForestRegressor(random_state=40),param_grid,cv=5,verbose=0,n_jobs=-1,scoring='neg_root_mean_squared_error')
    rf_mod_cv.fit(train_X_sc,train_Y)
    rf_mod=rf_mod_cv.best_estimator_
    prediction_file=test.loc[:,['Item_Identifier','Outlet_Identifier']]
    prediction_file['Item_Outlet_Sales']=rf_mod.predict(prediction_X_sc)
    file_name=feature_engg_method+'_rf_prediction'
    prediction_file.to_csv(f'C:/Users/sidha/OneDrive/Desktop/Work/Hckthon/ABB/Output/{file_name}.csv',index=False)

    # XGBoost Regressor
    param_grid=[{'n_estimators': [100,500,1000], 'learning_rate':[0.001,0.01,0.1],'subsample':[0.8]}]
    xgb_mod_cv=GridSearchCV(XGBRegressor(random_state=40),param_grid,cv=5,verbose=0,n_jobs=-1,scoring='neg_root_mean_squared_error')
    xgb_mod_cv.fit(train_X_sc,train_Y)
    xgb_mod=xgb_mod_cv.best_estimator_
    prediction_file=test.loc[:,['Item_Identifier','Outlet_Identifier']]
    prediction_file['Item_Outlet_Sales']=xgb_mod.predict(prediction_X_sc)
    file_name=feature_engg_method+'_xgb_prediction_v_'+str(version_number)
    prediction_file.to_csv(f'C:/Users/sidha/OneDrive/Desktop/Work/Hckthon/ABB/Output/{file_name}.csv',index=False)

    # LightGBM Regressor
    param_grid=[{'n_estimators': [100, 150, 200, 250, 300], 'learning_rate':[0.001,0.01,0.1],'subsample':[0.8]}]
    lgb_mod_cv=GridSearchCV(LGBMRegressor(random_state=40),param_grid,cv=5,verbose=0,n_jobs=-1,scoring='neg_root_mean_squared_error')
    lgb_mod_cv.fit(train_X_sc,train_Y)
    lgb_mod=lgb_mod_cv.best_estimator_
    prediction_file=test.loc[:,['Item_Identifier','Outlet_Identifier']]
    prediction_file['Item_Outlet_Sales']=lgb_mod.predict(prediction_X_sc)
    file_name=feature_engg_method+'_lgbm_prediction_v_'+str(version_number)
    prediction_file.to_csv(f'C:/Users/sidha/OneDrive/Desktop/Work/Hckthon/ABB/Output/{file_name}.csv',index=False)

    # CatBoost Regressor
    cb_mod=CatBoostRegressor(random_state=42,verbose=0)
    cb_mod.fit(train_X_sc,train_Y,eval_set=(test_X_sc,test_Y),early_stopping_rounds=1)
    prediction_file=test.loc[:,['Item_Identifier','Outlet_Identifier']]
    prediction_file['Item_Outlet_Sales']=cb_mod.predict(prediction_X_sc)
    file_name=feature_engg_method+'_cb_prediction_v_'+str(version_number)
    prediction_file.to_csv(f'C:/Users/sidha/OneDrive/Desktop/Work/Hckthon/ABB/Output/{file_name}.csv',index=False)

    # KNN Regressor 
    param_grid=[{'n_neighbors': [5,10,15,20,25,30]}]
    knn_mod_cv=GridSearchCV(KNeighborsRegressor(),param_grid,cv=5,verbose=0,n_jobs=-1,scoring='neg_root_mean_squared_error')
    knn_mod_cv.fit(train_X_sc,train_Y)
    knn_mod=knn_mod_cv.best_estimator_
    prediction_file=test.loc[:,['Item_Identifier','Outlet_Identifier']]
    prediction_file['Item_Outlet_Sales']=knn_mod.predict(prediction_X_sc)
    file_name=feature_engg_method+'_knn_prediction_v_'+str(version_number)
    prediction_file.to_csv(f'C:/Users/sidha/OneDrive/Desktop/Work/Hckthon/ABB/Output/{file_name}.csv',index=False)

    # SVM Regressor
    param_grid=[{'C': [0.1, 1, 100, 1000], 'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]}]
    svm_mod_cv=GridSearchCV(SVR(),param_grid,cv=5,verbose=0,n_jobs=-1,scoring='neg_root_mean_squared_error')
    svm_mod_cv.fit(train_X_sc,train_Y)
    svm_mod=svm_mod_cv.best_estimator_
    prediction_file=test.loc[:,['Item_Identifier','Outlet_Identifier']]
    prediction_file['Item_Outlet_Sales']=svm_mod.predict(prediction_X_sc)
    file_name=feature_engg_method+'_svm_prediction_v_'+str(version_number)
    prediction_file.to_csv(f'C:/Users/sidha/OneDrive/Desktop/Work/Hckthon/ABB/Output/{file_name}.csv',index=False)

