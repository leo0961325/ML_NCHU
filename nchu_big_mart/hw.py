import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder ,Normalizer
from sklearn.linear_model import RANSACRegressor ,Ridge ,RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.feature_selection import SelectFromModel 
from sklearn.model_selection import cross_val_score 




test = pd.read_csv('nchu_big_mart/test.csv')
train = pd.read_csv('nchu_big_mart/train.csv')

data = train.append(test, ignore_index = True) #combined data

def missing_and_desc():
    missing_data = data.apply(lambda x: sum(x.isnull()))
    print(missing_data)
    desc = train.describe() 
    print(desc)

#we found Item_Weight , Outlet_Size_count ,Item_Outlet_Sales has missing value
Outlet_Size_count = data['Outlet_Size'].value_counts()
print(Outlet_Size_count)
data.Outlet_Size = data.Outlet_Size.fillna('Medium')
#outlet size use most frequecy "Medium" to fill.

data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())
data.Item_Outlet_Sales = data.Item_Outlet_Sales.fillna(data.Item_Outlet_Sales.mean())

data['Item_Weight'] = data['Item_Weight'].replace(0, np.NaN)
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace = True)

data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace = True)

data['Item_Outlet_Sales'] = data['Item_Outlet_Sales'].replace(0, np.NaN)
data['Item_Outlet_Sales'].fillna(data['Item_Outlet_Sales'].mode()[0], inplace = True)

data.isnull().sum()

# 替代 reg -> Regular ; Low Fat, low fat and, LF

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
data['Item_Fat_Content'].value_counts()

# 取前兩個字，作為簡化

data['Item_Identifier'] = data['Item_Identifier'].apply(lambda x: x[0:2])

data['Item_Identifier'] = data['Item_Identifier'].map({'FD':'Food', 'NC':'Non_Consumable', 'DR':'Drinks'})

data['Item_Identifier'].value_counts()


#  使用label encoding轉化


data.apply(LabelEncoder().fit_transform)
data = pd.get_dummies(data)


# 將 dataset 分割成訓練集合測試集 train and test

train = data.iloc[:8523,:]
test = data.iloc[8523:,:]

X = train.drop('Item_Outlet_Sales', axis = 1)
Y = train.Item_Outlet_Sales
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size =0.2)

print(data)
#LR 
regressor = LinearRegression()
regressor.fit(X_train, y_train)
lr = LinearRegression(normalize=True)
lr.fit(X_train,y_train)
lr_score= lr.score(X_test, y_test)
print("LR : " , lr_score)

#LR with cross_val_score
lr_scores_cv = cross_val_score(lr, X,Y, cv=10, scoring='r2')
print ("LR with cv :" , lr_scores_cv.mean())

#RANSAC_Regressor RS
lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
rs = RANSACRegressor(lr)
rs.fit(X, Y)
rs.score(X, Y)
rs.estimator_.intercept_
rs.estimator_.coef_

rs_score = rs.score(X,Y)
rs_score = cross_val_score(rs, X,Y, cv=10, scoring='r2')
print("RS: " ,rs_score.mean())



#Ridge_and_Lasso -> RG
rg = Ridge(alpha=0.001, normalize=True)
rg = RidgeCV(alphas=(1.0, 0.1, 0.01, 0.005, 0.0025, 0.001, 0.00025), normalize=True)
rg.fit(X, Y)
rg_scores = cross_val_score(rg, X, Y, cv=10, scoring='r2')
score = rg_scores.mean()
print("RG : " ,score)
print(rg.alpha_) # the suitble is 0.005

#ElasticNet

en = ElasticNet(alpha=0.001, l1_ratio=0.8, normalize=True)    
en_scores = cross_val_score(en,X,Y, cv=10, scoring='r2')
encv = ElasticNetCV(alphas=(0.1, 0.01, 0.005, 0.0025, 0.001), l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.8), normalize=True)
encv.fit(X, Y)
print("EN : " , en_scores.mean())




# PolynomialFeatures
pf = PolynomialFeatures(degree=2)
Xp = pf.fit_transform(X)
lr = LinearRegression(normalize=True)
lr.fit(Xp, Y)
score = lr.score(Xp, Y)
print("PF : " ,score)
