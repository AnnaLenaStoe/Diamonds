import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import math

diamond = pd.read_csv('diamonds.csv')

diamond.head()

diamond.info()

diamond.describe()

diamond.columns
diamond.drop(['Unnamed: 0'] , axis=1 , inplace=True)
diamond.describe ()

sns.pairplot(diamond)

sns.distplot(diamond['price'])

sns.heatmap(diamond.corr())

#sns.factorplot(data=diamond , kind='box' , size=7, aspect=2.5)

#sns.kdeplot(diamond['carat'], shade=True , color='b')

#sns.jointplot(x='carat' , y='price' , data=diamond , size=5)

#sns.factorplot(x='cut', data=diamond , kind='count',aspect=2.5 )

#sns.factorplot(x='cut', y='price', data=diamond, kind='box' ,aspect=2.5 )

#sns.factorplot(x='color', data=diamond , kind='count',aspect=2.5 )

#sns.factorplot(x='color', y='price' , data=diamond , kind='violin', aspect=2.5)

#sns.boxplot(x='clarity', y='price', data=diamond)

#plt.hist('depth' , data=diamond , bins=25)

#sns.jointplot(x='depth', y='price' , data=diamond , kind='regplot', size=5)

#sns.kdeplot(diamond['table'] ,shade=True , color='b')

x = diamond[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y = diamond['price']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

coeff_diamond = pd.DataFrame(lr.coef_,x.columns,columns=['Coefficient'])
coeff_diamond

predictions = lr.predict(x_test)

plt.scatter(y_test,predictions)

sns.distplot((y_test-predictions),bins=50);

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

