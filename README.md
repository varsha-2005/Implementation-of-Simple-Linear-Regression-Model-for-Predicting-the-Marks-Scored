# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE.



## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: varsha.g
RegisterNumber: 22002003
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()


df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,regressor.predict(x_test),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
![v1](https://user-images.githubusercontent.com/119288183/229331075-16a22235-cdad-45f0-97a9-1f21009a2699.png)


![tail](https://user-images.githubusercontent.com/119288183/229331098-05926dc6-cab8-46aa-bf4a-f0e571d3d4d6.png)

![x](https://user-images.githubusercontent.com/119288183/229331182-4d6b82fd-f8ab-4ec5-80f3-6221e60b977e.png)


![y](https://user-images.githubusercontent.com/119288183/229331167-82d991a7-47b6-48e5-833c-33ce1b3dc88d.png)

![ypredct](https://user-images.githubusercontent.com/119288183/229331206-1cbe9d83-bc81-495c-ba48-49917c0ec901.png)

![ytest](https://user-images.githubusercontent.com/119288183/229331224-4d72bb36-e9e4-44aa-8ae1-8f197de45330.png)

![graph1](https://user-images.githubusercontent.com/119288183/229331243-7cb022e6-9619-4ce7-a181-7f579667dc18.png)

![graph2](https://user-images.githubusercontent.com/119288183/229331248-b6adebf0-5a69-4513-9bf1-956b1a1c8568.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
