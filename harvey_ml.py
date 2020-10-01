import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
import numpy as np



#delimiter = 定界符
#header = 指定第幾行作為列名 默認infer

data = pd.read_csv('bank.csv' , delimiter = ';' , header ='infer')
data_set = data.head()
#print(data_set)

# data.balance.hist()
# plt.title('Histogram of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()


final = data.drop(['job','education','contact','day','default' , 'month','duration','pdays','previous','marital'],axis=1)
final.y.replace(('yes' , 'no') , (1,0), inplace =True)
final.loan.replace(('yes' , 'no') , (1,0) , inplace = True)
final.housing.replace(('yes' , 'no') , (1,0) , inplace = True)
#final.marital.replace(("married","divorced","single") , (2 ,1 , 0 ) , inplace = True)
final.poutcome.replace(("unknown","other","failure","success") , (0 , 1 , 2 ,3) , inplace = True)
data_set = final.head()
print(data_set)


x = final.drop(['y'] , axis = 1)
y = final.drop(['age' , 'balance','campaign' , 'housing' , 'loan' , 'poutcome' ] , axis = 1)
print(x.head())
#print(y.head())
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#classifer --- 線性回歸
lr = LinearRegression()
lr.fit(X_train  , y_train)
lr.score(X_test , y_test)
# print(lr.score(X_test , y_test))
# print(lr.coef_)

#classifer --- 決策樹
dt1 = tree.DecisionTreeClassifier(max_depth=4) #調整深度
dt1.fit(X_train, y_train)
dt1.score(X_test, y_test)
print(dt1.score(X_test, y_test))


pred = dt1.predict(np.array([30, 6000, 0 , 0 , 5 ,0]).reshape(1,-1))
print(pred)
