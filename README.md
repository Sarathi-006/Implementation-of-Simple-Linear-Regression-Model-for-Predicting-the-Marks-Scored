# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and inspect the first and last few rows.

2.Split the data into feature (x) and target (y), assuming the last column is the target.

3.Split the data into training and testing sets using train_test_split().

4.Train a linear regression model using the training data (x_train, y_train).

5.Predict the target values for the test data (x_test) using the trained model.

6.Visualize the training set and test set by plotting scatter plots and regression lines.

7.Evaluate the model using metrics: MSE, MAE, and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.PARTHASARATHI
RegisterNumber:  22223040144
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
DATASETS:
![307611951-9ce613a7-871c-42a8-9eff-b83ace99b155](https://github.com/user-attachments/assets/9f276c89-4dce-4cb9-b3c7-dc0949458daa)
HEADVALUES:
![307612004-345b1045-a2b4-4b4e-9945-0986a18926ce](https://github.com/user-attachments/assets/59fff252-1da1-4898-a9a1-54e181e82ad6)
TAILVALUES:
![307612029-a60a945b-5202-4793-88fe-e38b4961d0fc](https://github.com/user-attachments/assets/75364f55-0fae-4a74-99ef-7ff9de9331b0)
X AND Y VALUES:
![307612053-c2f170e9-eb13-4751-adc2-6902066cdeda](https://github.com/user-attachments/assets/21967c80-650c-46be-b656-c20a19330adb)
PREDICTION VALUES OF X AND Y:
![307612078-f6272467-aacc-4dd5-956c-0ddea06fcf37](https://github.com/user-attachments/assets/44c56a19-f1bb-4cfb-8756-58239c118b12)
MSE,MAE AND RMSE:
![307612152-78032443-3a6f-45f5-9683-a324d7f7d44d](https://github.com/user-attachments/assets/e93d9276-1d2d-4d19-893c-e5142e8d540d)
TRAINING SETS:
![307612165-d73d872e-db20-4fbe-a396-c8bcdea2b373](https://github.com/user-attachments/assets/ce5d3fec-41e8-44fd-9a9d-8e8873b0a02a)
TRAINING SETS:
![307612194-179d98ab-8a49-4b48-9b4d-6e7e2b2ede7a](https://github.com/user-attachments/assets/0e1d212a-376b-4c68-b70c-655fe1e3bc34)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
