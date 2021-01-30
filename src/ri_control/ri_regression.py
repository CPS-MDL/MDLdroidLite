import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

scale = 100.


def linear_regression(x, y):
    lr = LinearRegression().fit(x, y)
    return lr


def svr_regression(x, y, kernel='linear'):
    svr = SVR(kernel=kernel, C=100, gamma=0.1, epsilon=.1)
    svr.fit(x, y)

    return svr


def func(x, a, b):
    return a * np.power(b, x)


def funcb(x, a):
    return a * np.power(0.98, x)


# mnist
def fit_decay(xdata, ydata):
    # bounds = (0, [scale, 1.])
    bounds = (0, [scale, 0.99999999])
    # bounds = (0, [scale, 0.99])

    popt, pcov = curve_fit(func, xdata, ydata, bounds=bounds)
    print(popt)

    def curve(x):
        return func(x, *popt)

    return curve, popt


# change a's bound to 100-c
def fit_decay_constant(xdata, ydata, c):
    def fun_cons(x, a, b):
        return a * np.power(b, x) + c

    bounds = (0, [100 - c, 1])
    popt, pcov = curve_fit(fun_cons, xdata, ydata, bounds=bounds)
    print('P(a,b): {} , P(c): {}'.format(popt, c))

    def curve(x):
        return fun_cons(x, *popt)

    return curve


# def fit_decay(xdata, ydata):
#     # bounds = (0, [1., 1.])
#     bounds = ([0, 0, scale / 2], [scale, 1, scale])
#
#     popt, pcov = curve_fit(func, xdata, ydata, bounds=bounds)
#     print(popt)
#
#     def curve(x):
#         return func(x, *popt)
#
#     return curve


def fit_decay_old(xdata, ydata):
    bounds = (0, [100., 1., 1.])

    popt, pcov = curve_fit(funcd, xdata, ydata, bounds=bounds)
    print(popt)

    def curve(x):
        return funcd(x, *popt)

    return curve


def funcd(x, a, b, c):
    return a * np.power(b, x) + c


# minist
def fit_self_decay(xdata, ydata):
    # bounds = (0, [scale, 0.99, scale / 10])
    # bounds = (0, [scale, 1, scale/10])
    bounds = (0, [scale, 0.99999999, scale/10])
    # bounds = (0, [2., 1., 2.])

    popt, pcov = curve_fit(funcd, xdata, ydata, bounds=bounds)
    print(popt)

    def curve(x):
        return funcd(x, *popt)

    return curve, popt


# delete c har
# def fit_self_decay(xdata, ydata):
#     bounds = (0, [scale, 1])
#     # bounds = (0, [scale, 1, scale/10])
#     # bounds = (0, [2., 1., 2.])
#
#     popt, pcov = curve_fit(func, xdata, ydata, bounds=bounds)
#     print(popt)
#
#     def curve(x):
#         return func(x, *popt)
#
#     return curve


def fit_decay_short(xdata, ydata):
    bounds = (0, [10.])

    popt, pcov = curve_fit(funcb, xdata, ydata, bounds=bounds)

    # print(popt)

    def curve(x):
        return funcb(x, *popt)

    return curve


if __name__ == "__main__":
    # import numpy as np
    # from sklearn.linear_model import LinearRegression
    # 
    # X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # # y = 1 * x_0 + 2 * x_1 + 3
    # y = np.dot(X, np.array([1, 2])) + 3
    # reg = LinearRegression().fit(X, y)
    # reg.score(X, y)
    # 
    # reg.predict(np.array([[3, 5]]))
    x = np.asarray([2, 3, 7])
    y = np.asarray([8, 7, 5])
    curve = fit_decay(x, y)
    print(curve(x))
    print(curve(10))
