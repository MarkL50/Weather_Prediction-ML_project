# ML_Project_GDSC BIT_Meshra
## Machine Learning Model for weather prediction and error predeiction using tensorflow.

This machine learning model has been trained to read the dataset (csv file) provided by the user and present the graph for the temperature , precipitation , humidity etc.
In the second section , it will also be trained to detect the errors  in the prediction of central tendencies. So let's see how the code actually works:-

## 1 .First of all we need to import necessary libraries for different purposes. 

import numpy as np  # linear algebra
import pandas as pd # data processing(CSV file)
import matplotlib.pyplot as plt  #generating graphs

## 2 .User will have to upload the dataset mannually in the form of csv file and the input the code to read the dataset 

( The csv file and parameters such as  temperature , precipaitation etc )

## 3 .The following code will provide the preview of the dataset that was given as input . 
dataset.head(10)

## OUTPUT: 

Date |  temperaturemin | temperaturemax | precipitation | snowfall | snowdepth	|
--- | --- | --- | --- |--- |--- |
2021-02-13 |  25.0 |  61.0 |  0.00 | 286 | 289 | 
--- | --- | --- | --- |--- |--- |--- |--- |--- |
2021-02-16 |  34.0 |  63.0 |  0.00 |  0.0 |  0.0 |  
--- | --- | --- | --- |--- |--- |--- |--- |--- |
2021-02-18 |  52.0 |  78.1 |  0.00 |  0.0 |  0.0 |  
--- | --- | --- | --- |--- |--- |--- |--- |--- |
2021-03-03 |  35.1 |  53.1 |  0.00 |  0.0 |  0.0 |  
--- | --- | --- | --- |--- |--- |--- |--- |--- |

    date temperaturemin temperaturemax precipitation snowfall snowdepth	
   2021-02-13	    25.0	        61.0	            0.00	       0.0	      0.0	
   
   2021-02-16	    34.0	        63.0	            0.00	       0.0	      0.0
   
   2021-02-18	    52.0	        78.1	            0.00	       0.0	      0.0	
   
   2021-03-03	    35.1	        53.1	            0.00	       0.0	      0.0	
    So on .........
## 4 .The following function with rehshape the array.
x = np.array(temp).reshape(-1, 1) # function name says it : reshape the array
y = np.array(precipitation)

## 5 .Then comes the following command that is used to split arrays or matrices into random train and test subsets so that we don't have to segregate them manually.
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size=1/3, random_state=0 )
(train_test_split will make random partitions for the two subsets)

## 6 .from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit( xtrain, ytrain )
(to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.)

## regressor.coef_ , regressor.intercept_# (y = mx + c , m is coef , c is intercept)

## .Root mean squared error (RMSE) : If True returns MSE value, if False returns RMSE value.

np.sqrt ( sum( abs( actualValue** 2 - predictedValue** 2 ) ) ) / len( xtrain ) # RMSE

## .Now after the completion of training and testing of the model , we need to show the graph for prediction of training dataset .
plt.scatter(xtrain, ytrain, color='cyan') (# x = xtrain , y = ytrain)

#Predicted values
prediction = regressor.predict(xtrain)
plt.plot(xtrain, prediction , color = 'black') (# y = prediction)

plt.title ("Prediction for Training Dataset")
plt.xlabel("Temperature in degree"), plt.ylabel("Precipitation")
plt.show()

![manny12](https://user-images.githubusercontent.com/76861726/152522044-4238b9d0-bdd6-481f-9737-11410d6243a6.png)

## The following graph will show the Final training dataset :

plt.scatter(xtest, ytest, color= 'green')

plt.plot(xtrain, regressor.predict(xtrain), color = 'black')

plt.title ("Training Dataset")
plt.xlabel("Tempertaure in degree"), plt.ylabel("Precipitation")
plt.show()


![manny23](https://user-images.githubusercontent.com/76861726/152522238-d00ea81f-5712-4350-bd2a-35c461975574.png)

## Code snippet to show the graph of the windspeed . 

d=dataset['avgwindspeed'].value_counts()
d.plot(kind='bar')

![Screenshot (570)](https://user-images.githubusercontent.com/76861726/152522735-283e7f89-c933-42fc-b60e-5a76871d7d06.png)





