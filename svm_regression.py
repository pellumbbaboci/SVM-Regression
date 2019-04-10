#Support Vector Regression using linear and non-linear kernels
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()


y[::5] += 3 * (0.5 - np.random.rand(8))# Add noise to targets


# Fit regression model
sv_regression_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
sv_regression_lin = SVR(kernel='linear', C=1e3)
sv_regression_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = sv_regression_rbf.fit(X, y).predict(X)
y_linear = sv_regression_lin.fit(X, y).predict(X)
y_polynomial = sv_regression_poly.fit(X, y).predict(X)


lw = 2
plt.scatter(X, y, color='r', label='data')
plt.plot(X, y_rbf, color='g', lw=lw, label='RBF model')
plt.plot(X, y_linear, color='c', lw=lw, label='Linear model')
plt.plot(X, y_polynomial, color='cornflowerblue', lw=lw, label='Polynomial model')

plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
