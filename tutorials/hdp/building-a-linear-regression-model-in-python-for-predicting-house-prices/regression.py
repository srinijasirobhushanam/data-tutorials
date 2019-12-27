import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

boston = load_boston()
boston.keys()
print (boston.DESCR) # Gives the clear information about the columns

boston.data.shape
print (boston.feature_names)

boston_dataset = pd.DataFrame(boston.data)
boston_dataset.columns = boston.feature_names
boston_dataset.head()

boston_dataset['PRICE'] = boston.target
boston_dataset.head()

boston_dataset.describe()

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from matplotlib import rcParams
plt.scatter(boston_dataset.CRIM, boston_dataset.PRICE)
plt.xlabel("Per capita crime rate by town")
plt.ylabel("Price of the house")
plt.title("Relationship between crime rate and Price")

plt.scatter(boston_dataset.RM, boston_dataset.PRICE)
plt.xlabel("Average number of rooms per dwelling")
plt.ylabel("Price of the house")
plt.title("Relationship between rooms per dwelling and Price")


plt.scatter(boston_dataset.PTRATIO, boston_dataset.PRICE)
plt.xlabel("Pupil-teacher ratio by town")
plt.ylabel("Price of the house")
plt.title("Relationship between PTRATIO and Price")


from sklearn.linear_model import LinearRegression
X = boston_dataset.RM
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    X, boston_dataset.PRICE, test_size=0.2, random_state = 5)
print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)
x_train_new= X_train.reshape(-1,1)
y_train_new= Y_train.reshape(-1,1)
x_test_new= X_test.reshape(-1,1)
y_test_new= Y_test.reshape(-1,1)

linear_model = LinearRegression()

linear_model.fit(x_train_new,y_train_new)

print ('Estimated intercept coefficient:', linear_model.intercept_)

pred = linear_model.predict(x_test_new)

print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - y_test_new)))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - y_test_new) ** 2))
print("R2-score: %.2f" % r2_score(pred , y_test_new) )

X = boston_dataset.drop('PRICE', axis = 1)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
X, boston_dataset.PRICE, test_size=0.2, random_state = 5)
# This creates a LinearRegression object
linear_model = LinearRegression()

linear_model.fit(X_train,Y_train)
pred = linear_model.predict(X_test)

print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - Y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - Y_test) ** 2))
print("R2-score: %.2f" % r2_score(pred , Y_test) )

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt

my_data = genfromtxt('Salary_Data.csv', delimiter=',',skip_header=1 )
X = np.c_[np.ones(my_data.shape[0]),my_data[:,0]]
y = np.c_[my_data[:,1]]
m = y.size 

def h(beta,X): 
    return np.dot(X,beta)
  
def computeCost(val_beta,X,y): 
    return float((1./(2*m)) * np.dot((h(val_beta,X)-y).T,(h(val_beta,X)-y)))

initial_beta = np.zeros((X.shape[1],1))
print (computeCost(initial_beta,X,y))

iterations = 500
lr= 0.01
#Initial parameters are set to 0.
iterations= 500
alpha= 0.01
def descendGradient(X, beta_start = np.zeros(2)):
    beta = beta_start
    costvec = [] #Used to plot cost as function of iteration
    betavalues = [] 
    for val in range(iterations):
        tmpbeta = beta
        costvec.append(computeCost(beta,X,y))
        betavalues.append(list(beta[:,0]))
        #Simultaneously updating theta values
        for j in range(len(tmpbeta)):
            tmpbeta[j] = beta[j] - (alpha/m)*np.sum((h(beta,X) - y)*np.array(X[:,j]).reshape(m,1))
        beta = tmpbeta
    return beta, betavalues, costvec
initial_beta = np.zeros((X.shape[1],1))
beta, betavalues, costvec = descendGradient(X,initial_beta)


def fit(xval):
    return beta[0] + beta[1]*xval
plt.figure(figsize=(10,6))
plt.plot(X[:,1],y[:,0],'rx',markersize=10,label='Training Data')
plt.plot(X[:,1],fit(X[:,1]),'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(beta[0],beta[1]))
plt.grid(True)
plt.ylabel('Years of experience')
plt.xlabel('Salary')
plt.legend()

from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools
fig = plt.figure(figsize=(20,20))
ax = fig.gca(projection='3d')
xvals = np.arange(-10,10,.5)
yvals = np.arange(-1,4,.1)
xs, ys, zs = [], [], []
for a in xvals:
    for b in yvals:
        xs.append(a)
        ys.append(b)
        zs.append(computeCost(np.array([[a], [b]]),X,y))
scat = ax.scatter(xs,ys,zs,c=np.abs(zs),cmap=plt.get_cmap('YlOrRd'))
plt.xlabel(r'$\beta_0$',fontsize=30)
plt.ylabel(r'$\beta_1$',fontsize=30)
plt.title('Cost Minimization Path',fontsize=30)
plt.plot([x[0] for x in betavalues],[x[1] for x in betavalues],costvec,'bo-')
plt.show()

