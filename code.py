import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
############## data processing   
from sklearn.model_selection import train_test_split
############ algorithms 
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
#### accuracy scoring 
from sklearn.metrics import mean_squared_error 

# load data

train_data =pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/train.csv') 
test_data = pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/test.csv')
sample = pd.read_csv('../input/smart-homes-temperature-time-series-forecasting/sample_submission.csv')

# show data
train_data.drop(['Id' ] , axis=1 ,inplace=True) 
test_data.drop(['Id'] , axis=1 , inplace=True)  


train_data.head()
test_data.head(5)


#show data type of Date and Time
print(np.dtype(train_data['Date'])) 
print(np.dtype(train_data['Time']))
print("*************************")
print(np.dtype(test_data['Date'])) 
print(np.dtype(test_data['Time']))

#check for nulls values 
print('nul values in train data',sum(train_data.isna().sum()))
print('nul values in test data ' ,sum(test_data.isna().sum()))

# data processing
train_data['Date'] = pd.to_datetime(train_data['Date'] , format ="%d/%m/%Y")
train_data['Date'] = train_data['Date'].dt.dayofyear.astype(float)
test_data['Date'] = pd.to_datetime(test_data['Date'] , format = '%d/%m/%Y')  
test_data['Date'] = test_data['Date'].dt.dayofyear.astype(float)


train_data['Time'] = pd.DatetimeIndex(train_data['Time'])
train_data['Minutes'] = train_data['Time'].apply(lambda x :x.hour *60 + x.minute).astype(float)
test_data['Time'] = pd.DatetimeIndex(test_data['Time'])
test_data['Minutes']  =test_data['Time'].apply(lambda x : x.hour *60 + x.minute).astype(float)


plt.figure(figsize=(10,10))
sns.histplot(train_data['Time'])


y=train_data['Indoor_temperature_room']
plt.figure(figsize=(15,15) , facecolor='red')
plt.scatter(x=train_data['Day_of_the_week'] , y=y , marker='*')


train_data['day_of_the_year'] = train_data['Date']
test_data['day_of_the_year'] = test_data['Date']


train_data = train_data[(train_data.day_of_the_year != 73) 
                          & (train_data.day_of_the_year != 80) 
                          & (train_data.day_of_the_year != 309)]


#  feature Engineering

# we will add some related columns with CO2



for data in (train_data, test_data):
    data["CO2_avg"] = (data["CO2_(dinning-room)"] + data["CO2_room"])/2
    data["CO2_avg_and_shift"] = data["CO2_avg"].shift(4).bfill()
    data['CO2_avg_mean'] = data.groupby('day_of_the_year')['CO2_avg'].transform('mean')
    
    
train_data = train_data.drop(['Day_of_the_week' , 'Time'] , axis=1)

test_data = test_data.drop([ 'Day_of_the_week' , 'Time'] , axis=1)


train_data['Meteo_Rain'] = train_data['Meteo_Rain']*10 
test_data['Meteo_Rain'] = test_data['Meteo_Rain']*10

selection_feature = train_data.columns[3:].tolist()
selection_feature.remove('Indoor_temperature_room')


X_train = train_data[selection_feature]
y_train = train_data['Indoor_temperature_room']
X_test =  test_data[selection_feature]


###   use algoritms   ### 


XGB_model = XGBRegressor(base_score=0.5, booster='gblinear', colsample_bylevel=None,
             colsample_bynode=None, colsample_bytree=None, gamma=None,
             gpu_id=-1, importance_type='gain', interaction_constraints=None,
             learning_rate=0.15, max_delta_step=None, max_depth=15,
             min_child_weight=3, missing=np.nan, monotone_constraints=None,
             n_estimators=500, n_jobs=16, num_parallel_tree=None,
             random_state=0, reg_alpha=0, reg_lambda=0, scale_pos_weight=1,
             subsample=None, tree_method=None, validate_parameters=1,
             verbosity=None)



XGB_model.fit(X_train , y_train) 


# K-Nearest Neighbor

KNN_model = KNeighborsRegressor(n_neighbors=6, metric='euclidean')

KNN_model.fit(X_train , y_train) 


LGBM_model = LGBMRegressor(colsample_bytree=0.8,learning_rate=0.01, max_depth=8,
              min_child_weight=1, min_split_gain=0.0222415, n_estimators=35000,
              num_leaves=966, reg_alpha=0.04, reg_lambda=0.073,
              subsample=0.6)


LGBM_model.fit(X_train , y_train) 

test1_predict =XGB_model.predict(X_test) 
test1_predict[:20]


test2_predict =KNN_model.predict(X_test)
test2_predict[:20]

test3_predict =LGBM_model.predict(X_test)
test3_predict[:20]

























