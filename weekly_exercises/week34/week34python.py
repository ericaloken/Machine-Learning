import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#manual regression using linalg
x = np.random.rand(100,1)
y = 2.0+5*x*x+0.1*np.random.randn(100,1)

X = np.c_[np.ones((100,1)), x, x**2] 
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print(beta)

y_tilde = X @ beta

#Sklearn 
model = LinearRegression()
X_sklearn = np.c_[x, x**2]
model.fit(X_sklearn, y)

print(model.intercept_, model.coef_)

plt.scatter(x, y, label='Data')
plt.title('Manual regression using linalg vs sklearn')
index = np.argsort(x[:,0])
plt.plot(x[index], y_tilde[index], 'r-', label='Fitted')

plt.plot(x[index], model.predict(X_sklearn)[index], 'g--', label='Sklearn Fitted')
plt.legend()
plt.show()