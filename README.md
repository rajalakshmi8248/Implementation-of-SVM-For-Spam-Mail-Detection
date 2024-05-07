# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RAJA LAKSHMI E
RegisterNumber: 212222220033
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Result Output
![282257583-78ccb346-ca7c-4a33-ad4c-e3355e1fddc6](https://github.com/rajalakshmi8248/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860827/ca94cad6-8891-4ed5-af53-078fb96eed93)

## data.head()
![282257589-139f19db-04ee-4e44-b04a-5f231988b90b](https://github.com/rajalakshmi8248/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860827/a01d0251-89ec-499c-a206-91cf1176a226)

## data.info()
![282257595-646ac557-8f21-442c-8783-6a1085ec89fd](https://github.com/rajalakshmi8248/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860827/a0046d3b-189b-4f08-a746-e7138e6d7542)

## data.isnull().sum()
![282257608-2eba109c-0bdd-468a-8bcf-d9258c23f8ef](https://github.com/rajalakshmi8248/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860827/8b44ab2d-25b4-4358-a2fb-9a308a06cfa8)
![282257631-b0e5dbc6-7c5b-40fe-b610-86fc97828918](https://github.com/rajalakshmi8248/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860827/6d711eef-c22f-497c-ad19-2b5a059dfaf6)

## Y_prediction Value
![282257837-6a911f5c-1e40-4047-9371-07a94f012cef](https://github.com/rajalakshmi8248/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860827/da04c4a8-6d43-4bb4-87b2-92ae5f0e20dc)

## Accuracy Value
![282257853-bad89364-aef2-4652-806d-09c5760c041e](https://github.com/rajalakshmi8248/Implementation-of-SVM-For-Spam-Mail-Detection/assets/122860827/ac251c4b-4169-45f3-823e-48728823aac1)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
