# ML homework : BigMarket  姓名 :　蔡閔珽 

> **本次以Big Mart Sales資料做資料預處理，再使用LinearRegression各項方法做分析，故分成兩部份，分別為資料處理及LR分析。**


### 1. Data preparation

> **Step 1.** 首先先將載好的test.csv和train.csv合併成一個檔案，並查看在 BigMarket data中，有哪些 missing values ，含有missing values共有三個分別是Item_Weight、Outlet_Size、Item_Outlet_Sales，於是先將Item_Weight和Item_Outlet_Sales先用mean將null補上，而Outlet_Size_count內容分別為Medium 4655個、Smaill 3980個、High 1553個，為了使用較好的數據
於是使用most frequency的Medium去填補Missing Value，完成後本次所選資料已無null值。
資料中"Item_Identifier"項目必須去做整合，將欄位中FDXXX、NCXXX、DRXXX等三種取前兩個字替換成FD':'Food','NC':'Non-Consumable','DR':'Drinks'將Item_Identifier整合。

> **Step 2.** 使用sklearn　preprocessing 中LabelEncoder, OneHotEncoder等兩個編碼方式將類別 (categorical)及文字(text)的資料轉換成數字，而讓程式能夠更好的去理解及運算，將feature各歸納成0101組成。

> **Step 3.** 最後資料預處理完成後再將一個檔案再分別存成train_mod.csv和test_mod.csv，方便下一步資料分析所用


### 2. Linear Regression

> 選用Linear Regression作為預測方法，而首先將分成將資料分成 train/test，而測試集選用其中20%做分析，將資料進行 normalize，再進行fit訓練，使用Linear_Regression分析出預測值為0.54-0.58之間，



