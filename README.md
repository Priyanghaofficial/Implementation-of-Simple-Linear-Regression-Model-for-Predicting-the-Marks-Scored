# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 
 

## Program:
```

Developed by:PRIYANGHA G 
RegisterNumber: 212223040157
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

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

# DATASET

![image](https://github.com/user-attachments/assets/086152da-3dd8-4966-b11d-0b75cfe21f03)

# HEAD VALUES

![image](https://github.com/user-attachments/assets/622b828b-eba6-4aef-a6aa-22e274a4f704)

# TAIL VALUES

![image](https://github.com/user-attachments/assets/72767627-4499-4524-88e3-090b1051a565)

# X and Y values 
![image](https://github.com/user-attachments/assets/b49303b5-0a6f-4897-b8b9-4583ac7eb5e7)
# Predication values of X and Y
![image](https://github.com/user-attachments/assets/e6a1246a-7836-4dd9-8de1-8fe3595efce1)
# TRAINING SET
![image](https://github.com/user-attachments/assets/9407cc8d-ca0e-4e20-ba2b-b605a5f703cc)
# TESTING SET AND MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/b4335ae9-f9b0-4884-ab62-2fbe37e4e54a)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
