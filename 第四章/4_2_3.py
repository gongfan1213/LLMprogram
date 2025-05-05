import numpy as np

m = 20
# 生成数据集x，此时的数据集x是一个二维矩阵
x0 = np.ones((m, 1))
x1 = np.arange(1, m+1).reshape(m, 1)
x = np.hstack((x0, x1)) #【20,2】
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)
alpha = 0.01

#这里的theta是一个[2,1]大小的矩阵，用来与输入x进行计算，获得计算的预测值y_pred，而y_pred是与y计算的误差
def error_function(theta,x,y):
    h_pred = np.dot(x,theta)
    j_theta = (1./2*m) * np.dot(np.transpose(h_pred), h_pred)
    return j_theta

def gradient_function(theta, X, y):
    h_pred = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), h_pred)

def gradient_descent(X, y, alpha):
    theta = np.array([1, 1]).reshape(2, 1)  #[2,1]  这里的theta是参数
    gradient = gradient_function(theta,X,y)
    while not np.all(np.absolute(gradient) <= 1e-6):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta

theta = gradient_descent(x, y, alpha)
print('optimal:', theta)
print('error function:', error_function(theta, x, y)[0,0])
