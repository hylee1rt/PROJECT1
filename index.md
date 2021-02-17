Hi there!

This project explores different machine learning models for univariate regression to predict the price of houses in Boston. We will be performing k-fold validation on all the regressors and comparing the average MAE of the mean absolute errors from the folds to evaluate their performance. 

```python

import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl

from google.colab import drive 
drive.mount('/content/gdrive')
df = pd.read_csv("gdrive/My Drive/Colab Notebooks/BostonHousingPrices.csv")

X = np.array(df['rooms']).reshape(-1,1)
y = np.array(df['cmedv']).reshape(-1,1)
dat = np.concatenate([X,y.reshape(-1,1)], axis=1)
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=2021)

y_train = y_train.reshape(len(y_train),)
y_test = y_test.reshape(len(y_test),)
dat_train = np.concatenate([X_train,y_train.reshape(-1,1)], axis=1)
dat_train = dat_train[np.argsort(dat_train[:, 0])]
dat_test = np.concatenate([X_test,y_test.reshape(-1,1)], axis=1)
dat_test = dat_test[np.argsort(dat_test[:, 0])]
```

### Linear Regression 

This classic model shows the linear relationship between number of rooms and the price of the house. The k-fold cross validation gives us the average of the mean absolute errors of each fold.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


kf = KFold(n_splits=10, shuffle=True, random_state=2021)
lm = LinearRegression() 

mae_lm = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  lm.fit(X_train.reshape(-1,1),y_train)
  yhat_lm = lm.predict(X_test.reshape(-1,1))
  mae_lm.append(mean_absolute_error(y_test, yhat_lm))
print("Validated MAE Linear Regression = ${:,.2f}".format(1000*np.mean(mae_lm)))

```

Validated MAE Linear Regression = $4,433.17

### Kernel Weighted Regressions

Locally weighted regresssions are non-parametric regression methods that combine multiple regression models in a k-nearest-neighbor-based meta-model. They are used to fit simple models to localized subsets of the data to build up a function that describes the variation in the data. Weights applied to each point help identify regions that contribute more heavily to the model.

The different kernels apply different weights to each point. We are looking for the model with the lowest average of mean absolute error obtained from k-fold validation. 

```python
def Tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)

def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 

def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2) 

def Cosine(x): 
  return np.where(np.abs(x)>1,0,(np.pi/4)*np.cos((np.pi/2)*np.radians(np.abs(x))))
```
```python
def lowess_kern(x, y, kern, tau):
  n = len(x)
  yest = np.zeros(n) 
  w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
  for i in range(n):
      weights = w[:, i]
      b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
      A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
      theta, res, rnk, s = linalg.lstsq(A, b)
      yest[i] = theta[0] + theta[1] * x[i] 

  return yest

def model_lowess(dat_train,dat_test,kern,tau):
  dat_train = dat_train[np.argsort(dat_train[:, 0])]
  dat_test = dat_test[np.argsort(dat_test[:, 0])]
  Yhat_lowess = lowess_kern(dat_train[:,0],dat_train[:,1],kern,tau)
  datl = np.concatenate([dat_train[:,0].reshape(-1,1),Yhat_lowess.reshape(-1,1)], axis=1)
  f = interp1d(datl[:,0], datl[:,1],fill_value='extrapolate')
  return f(dat_test[:,0])
```

```python
mae_lke = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lke = model_lowess(dat[idxtrain,:],dat[idxtest,:],Epanechnikov,0.45)
  mae_lke.append(mean_absolute_error(y_test, yhat_lke))
print("Validated MAE Local Epanechnikov Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lke)))
```
Validated MAE Local Epanechnikov Kernel Regression = $4,113.99

```python 
mae_lkt = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lkt = model_lowess(dat[idxtrain,:],dat[idxtest,:],Tricubic,0.45)
  mae_lkt.append(mean_absolute_error(y_test, yhat_lkt))
print("Validated MAE Local Tricubic Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lkt)))
```
Validated MAE Local Tricubic Kernel Regression = $4,110.37

```python
mae_lkq = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lkq = model_lowess(dat[idxtrain,:],dat[idxtest,:],Quartic,0.45)
  mae_lkq.append(mean_absolute_error(y_test, yhat_lkq))
print("Validated MAE Local Quartic Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lkq)))
```
Validated MAE Local Quartic Kernel Regression = $4,107.47

```python
mae_lkc = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lkc = model_lowess(dat[idxtrain,:],dat[idxtest,:],Cosine,0.45)
  mae_lkc.append(mean_absolute_error(y_test, yhat_lkc))
print("Validated MAE Local Quartic Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lkc)))
```
Validated MAE Local Cosine Kernel Regression = $4,125.13


### Random Forest

Random Forest is a classification model that consists of multiple, independent decision trees. 

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1000,max_depth=3)
mae_rf = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  rf.fit(X_train.reshape(-1,1),y_train)
  yhat_rf = rf.predict(X_test.reshape(-1,1))
  mae_rf.append(mean_absolute_error(y_test, yhat_rf))
print("Validated MAE RF = ${:,.2f}".format(1000*np.mean(mae_rf)))

```
Validated MAE Random Forest = $4,168.43

### Neural Networks

This model uses a network of functions, or layers of neurons, to understand and translate a data input of one form into a desired output. The neural network “learns” the data and fine-tunes the weights of the paths between these neurons to come up with accurate predictions. 

```python
# imports for creating a Neural Networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

```python
model = Sequential()
model.add(Dense(128, activation="relu", input_dim=1))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
mae_nn = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
  model.fit(X_train.reshape(-1,1),y_train,validation_split=0.3, epochs=1000, batch_size=100, verbose=0, callbacks=[es])
  yhat_nn = model.predict(X_test.reshape(-1,1))
  mae_nn.append(mean_absolute_error(y_test, yhat_nn))
print("Validated MAE Neural Network Regression = ${:,.2f}".format(1000*np.mean(mae_nn)))
```
Validated MAE Neural Network Regression = $4,260.39

### XGBoost (Extreme Gradient Boost)

This model is a decision-tree-based algorithm that uses an advanced implementation of gradient boosting and regularization framework for speed and performance. It can best be used to solve structured data such as regression, classification, ranking, and user-defined prediction problems. XGBoost focuses on minimizing the errors to turn weak learners into strong learners and "boost" performance.

```python
import xgboost as xgb
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
```
```python
mae_xgb = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model_xgb.fit(X_train.reshape(-1,1),y_train)
  yhat_xgb = model_xgb.predict(X_test.reshape(-1,1))
  mae_xgb.append(mean_absolute_error(y_test, yhat_xgb))
print("Validated MAE XGBoost Regression = ${:,.2f}".format(1000*np.mean(mae_xgb)))
```
Validated MAE XGBoost Regression = $4,136.63

### Support Vector Machine

SVMs work to find the best line (or hyperplane in n-dimensional space) that separates the data into separate classes. The objective is to find a hyperplane with the maximum margin - the maximum distance between data points of distinct classes. Support vectors are co-ordinates of individual data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. 

```python
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=4, epsilon=.1,coef0=1)
svr_sigm = SVR(kernel='sigmoid', C=100, gamma='auto', degree=4, epsilon=.1,coef0=1)
```

```python
model = svr_rbf
mae_svr = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model.fit(X_train.reshape(-1,1),y_train)
  yhat_svr = model.predict(X_test.reshape(-1,1))
  mae_svr.append(mean_absolute_error(y_test, yhat_svr))
print("Validated MAE Support Vector Regression = ${:,.2f}".format(1000*np.mean(mae_svr)))
```
Validated MAE Support Vector Regression = $4,130.50

```python
model = svr_lin
mae_svr = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model.fit(X_train.reshape(-1,1),y_train)
  yhat_svr = model.predict(X_test.reshape(-1,1))
  mae_svr.append(mean_absolute_error(y_test, yhat_svr))
print("Validated MAE Support Vector Regression = ${:,.2f}".format(1000*np.mean(mae_svr)))
```
Validated MAE Support Vector Regression = $4,432.00

*Poly kernel skipped because it takes so long..!*

```python
model = svr_sigm
mae_svr = []
for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model.fit(X_train.reshape(-1,1),y_train)
  yhat_svr = model.predict(X_test.reshape(-1,1))
  mae_svr.append(mean_absolute_error(y_test, yhat_svr))
print("Validated MAE Support Vector Regression = ${:,.2f}".format(1000*np.mean(mae_svr)))
```
Validated MAE Support Vector Regression = $6,540.67

*...oof.*

### Conclusion 

The locally weighted regression (LOWESS) with Quartic kernel had the lowest mean absolute error of $4,107.47, performing the best (even better than neural network!).


