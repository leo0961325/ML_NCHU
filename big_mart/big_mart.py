import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings




# step1. Combine test and train into one file
train = pd.read_csv("big_mart/train.csv")
test = pd.read_csv("big_mart/test.csv")


train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
#print(train.shape, test.shape, data.shape) #that will be (8523, 13) (5681, 12) (14204, 13)
#print(data.describe())

# step2. Check missing values:
check_missing_data = data.apply(lambda x: sum(x.isnull()))
#print(check_missing_data) #-> missing feature : (Item_Weight :2439) (Outlet_Size:4016) (Item_Outlet_Sales :5681)

# step3. Filling missing values

#print(data.head()) 

data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())
data.Item_Outlet_Sales = data.Item_Outlet_Sales.fillna(data.Item_Outlet_Sales.mean())

Outlet_Size_count = data['Outlet_Size'].value_counts()
print(Outlet_Size_count) #Find out the three types of Outlet_Size_count: (Medium 4655) (Small 3980) (High 1553)
data.Outlet_Size = data.Outlet_Size.fillna('Medium')

# -> Check whether the data has been cleaned
check_missing_data2 = data.apply(lambda x: sum(x.isnull()))
#print(check_missing_data2 , data.info())  #data.info to check Non-Null Count

# step3-1 #Item type combine:
data['Item_Identifier'].value_counts()

# Divide the first two words of Item_Identifier into three types:'Food' 'Non-Consumable' 'Drinks'  etc.
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2]) 
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'}) 
item_combined_data_counts = data['Item_Type_Combined'].value_counts()
#print(item_combined_data_counts) 



# step 4. One-Hot Coding of Categorical variables
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_modify = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_modify:
    data[i] = le.fit_transform(data[i])

data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
# Use One-Hot Coding
# print(data.head())

# USE warning :

def write_to_csv ():
    warnings.filterwarnings('ignore')
    #Drop the columns which have been converted to different types:
    data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

    #Divide into test and train:
    train = data.loc[data['source']=="train"]
    test = data.loc[data['source']=="test"]

    #Drop unnecessary columns:
    test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
    train.drop(['source'],axis=1,inplace=True)

    #Export files as modified versions:
    train.to_csv("big_mart/train_mod.csv",index=False)
    test.to_csv("big_mart/test_mod.csv",index=False)


write_to_csv()



