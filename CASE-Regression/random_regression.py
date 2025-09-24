# 回归分析
import random
from sklearn import linear_model

# 模拟构造一些数据
def generate(x):
	y = 2*x+10+random.random()
	return y

# 生成数据
train_x = []
train_y = []
for x in range(1000):
	train_x.append([x])
	y = generate(x)
	train_y.append([y])

reg = linear_model.LinearRegression()
# 模型训练 => 得到参数k,b
reg.fit(train_x, train_y)
# coef_ 保存线性模型的系数w
print(reg.coef_)
print(reg.intercept_)
# y = 1.99x + 10.53