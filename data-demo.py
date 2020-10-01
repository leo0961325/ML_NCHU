import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #訓練&測試
from sklearn.linear_model import LinearRegression #線性回歸
from sklearn import tree #決策樹
import numpy as np



#delimiter = 定界符
#header = 指定第幾行作為列名 默認infer

                                                                                          #1.讀取:

data = pd.read_csv('bank.csv' , delimiter = ';' , header ='infer')
data_set = data.head()
#print(data_set)

# data.balance.hist()                         #畫圖
# plt.title('Histogram of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()


final = data.drop(['job','marital','education','default','housing','contact','day','month','duration','campaign','pdays','previous','poutcome'],axis=1)
final.y.replace(('yes' , 'no') , (1,0), inplace =True)
final.loan.replace(('yes' , 'no') , (1,0) , inplace = True)
data_set = final.head()
#print(data_set)

                                                                                        #2.設置訓練

x = final.drop(['y'] , axis = 1)
y = final.drop(['age' , 'balance' , 'loan'] , axis = 1)
print(x.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

                                                                                        #3.使用classifer

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

                                                                                        #4.預測
pred = dt1.predict(np.array([40, 6000, 1]).reshape(1,-1))
print(pred)
