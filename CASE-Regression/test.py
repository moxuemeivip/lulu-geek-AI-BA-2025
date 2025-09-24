import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#TO DO：一元线性回归
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

# 创建线性回归模型实例
model = LinearRegression()

# 拟合模型
model.fit(x, y)

# 获取模型参数
r_sq = model.score(x, y)
print(f"决定系数(R^2): {r_sq}")

# 获取截距和斜率
print(f"截距: {model.intercept_}")
print(f"斜率: {model.coef_}")

# 进行预测
y_pred = model.predict(x)
print(f"预测值: {y_pred}")
print("========================================================================================")

#多元线性回归
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15],
[55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]

# 将列表转换为 numpy 数组
x = np.array(x)
y = np.array(y)

# 创建线性回归模型实例
model_multi = LinearRegression()

# 拟合多元线性回归模型
model_multi.fit(x, y)

# 获取模型的决定系数(R^2)
r_sq_multi = model_multi.score(x, y)
print(f"多元线性回归决定系数(R^2): {r_sq_multi}")

# 获取截距和斜率
print(f"多元线性回归截距: {model_multi.intercept_}")
print(f"多元线性回归斜率: {model_multi.coef_}")

# 进行预测
y_pred_multi = model_multi.predict(x)
print(f"多元线性回归预测值: {y_pred_multi}")
print("========================================================================================")

#多项式回归
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

# 构造多项式特征，包括x的0,1, 2阶特征
poly = PolynomialFeatures(degree=2, include_bias=True)
x_poly = poly.fit_transform(x)

# 创建线性回归模型实例
model_poly = LinearRegression()

# 拟合多项式回归模型
model_poly.fit(x_poly, y)

# 获取模型的决定系数(R^2)
r_sq_poly = model_poly.score(x_poly, y)
print(f"多项式回归决定系数(R^2): {r_sq_poly}")

# 获取截距和斜率
print(f"多项式回归截距: {model_poly.intercept_}")
print(f"多项式回归斜率: {model_poly.coef_}")

# 进行预测
y_pred_poly = model_poly.predict(x_poly)
print(f"多项式回归预测值: {y_pred_poly}")





