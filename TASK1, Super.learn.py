# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib.inline

data = pd.read_csv('http://bit.ly/w-data') 
# need a working connection to connectto the page
print('Data loaded successfully')
data.shape
data.head(6)
plt.plot(data) # visualizing the data.

''' Describing the data ''' 
data.describe()

#plotting the distribution
data.plot( x ='Hours', y = 'Scores', style = 'd')
plt.title('Hours Studied Vs Scored')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.legend()
plt.show()

''' The graph shows the data has a positive linear/
 relation between the two params. '''

X = data.iloc[:, :-1].values #Hours
y = data.iloc[:, 1:2].values #Scores

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg.score(X_train, y_train) # score method tells the accuracy of the model.
print(lin_reg.coef_) # The requisited slope
print(lin_reg.intercept_) # the intercept

''' The line will have the slope of [[9.71489047]] and will cut the y-aixs/
     at [3.42865287] '''

lor = lin_reg.coef_* X + lin_reg.intercept_ # REGRESSION LINE
plt.scatter(X, y)
plt.plot(X, lor, color = 'red')
plt.title(' Linear regression ')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()

print(X_test)
y_pred = lin_reg.predict(X_test) 

plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_test, y_pred, color = 'blue')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scored')
plt.show()

# Predicting the scores.
p = float(input("Enter the Hour studied"))
pred_scores = lin_reg.predict([[p]])
print("The predicted score is", pred_scores)














