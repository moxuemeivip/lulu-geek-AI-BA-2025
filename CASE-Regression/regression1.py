from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# 加载数据
diabetes = datasets.load_diabetes()
data = diabetes.data
print(data.shape)
print(data[0])

# 数据切分，训练集 70%，测试集 30%
train_x, test_x, train_y, test_y = train_test_split(data, diabetes.target, test_size=0.3)
print(len(train_x))

# 建立回归模型
clf = linear_model.LinearRegression()
# 学习系数k, b
clf.fit(train_x, train_y)
# coef_ 就是属性值， class_ 代表属性值
print(clf.coef_)
# 针对测试集进行预测
pred_y = clf.predict(test_x)
# 使用MSE进行评价
from sklearn.metrics import mean_squared_error
print(mean_squared_error(test_y, pred_y))
# 输出模型评分
r_sq = clf.score(train_x, train_y)
print('r_sq', r_sq)
