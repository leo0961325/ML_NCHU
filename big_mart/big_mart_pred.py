import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score #1.cross_val_score
from sklearn.preprocessing import Normalizer #2 Normalizer


# methods
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge #3.Ridge and Lasso 
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV #4.ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures #5.PolynomialFeatures
from sklearn.feature_selection import SelectFromModel #5



# Import the preprocessed files and divide them into training set and test set
train2 = pd.read_csv("big_mart/train_mod.csv")
test2 = pd.read_csv("big_mart/test_mod.csv")



X = train2.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis=1)
y = train2.Item_Outlet_Sales

# Predicting the test set results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



def Linear_Regression () :
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    

    lr_scores = lr.score(X_test, y_test)
    lr_scores = cross_val_score(lr, X,y, cv=16, scoring='r2') #cross_val_score

    print("Linear_Regression" ,lr_scores.mean())


def Linear_Regression_Normalizer () :
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    
    no = Normalizer()
    XN = no.fit_transform(X)

    lr_scores = lr.score(X_test, y_test)
    lr_scores = cross_val_score(lr, XN,y, cv=16, scoring='r2') #cross_val_score

    print("Linear_Regression_Normalizer : " ,lr_scores.mean())


def RANSAC_Regressor() :
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    rs = RANSACRegressor(lr)
    rs.fit(X, y)
    rs.score(X, y)
    rs.estimator_.intercept_
    rs.estimator_.coef_

    rs_score = rs.score(X,y)
    rs_score = cross_val_score(rs, X,y, cv=16, scoring='r2')
    print("RANSAC_Regressor : " ,rs_score.mean())



def Ridge_and_Lasso () :
    rg = Ridge(alpha=0.001, normalize=True)
    rg = RidgeCV(alphas=(1.0, 0.1, 0.01, 0.005, 0.0025, 0.001, 0.00025), normalize=True)
    rg.fit(X, y)
    rg_scores = cross_val_score(rg, X, y, cv=16, scoring='r2')
    pred = rg_scores.mean()
    print("Ridge_and_Lasso : " ,pred)
    #print(rg.alpha_) alpha適合值

def use_ElasticNet():
    en = ElasticNet(alpha=0.001, l1_ratio=0.8, normalize=True)    
    en_scores = cross_val_score(en,X, y, cv=16, scoring='r2')
    encv = ElasticNetCV(alphas=(0.1, 0.01, 0.005, 0.0025, 0.001), l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.8), normalize=True)
    encv.fit(X, y)
    print("ElasticNet : " , en_scores.mean())





def use_PolynomialFeatures():
    pf = PolynomialFeatures(degree=2)
    Xp = pf.fit_transform(X)
    lr = LinearRegression(normalize=True)
    lr.fit(Xp, y)
    score = lr.score(Xp, y)
    print("PolynomialFeatures : " ,score)
    
    sm = SelectFromModel(lr, threshold=10)
    Xt = sm.fit_transform(Xp, y) #shape
    score_mod = sm.estimator_.score(Xp, y)
    print("PolynomialFeatures in SelectFromModel :" ,score_mod)




def main () :

    Linear_Regression ()
    Linear_Regression_Normalizer ()
    RANSAC_Regressor()
    Ridge_and_Lasso ()
    use_ElasticNet()
    use_PolynomialFeatures()



main()


