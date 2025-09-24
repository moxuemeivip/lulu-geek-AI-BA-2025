import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.impute import SimpleImputer
import warnings
import sys
warnings.filterwarnings('ignore')

# 将输出重定向到文件
output_file = open('logistic_regression_results.txt', 'w', encoding='utf-8')
sys.stdout = output_file

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("="*80)
print("寿险客户续保预测 - 逻辑回归模型")
print("="*80)

# 1. 数据加载与探索
print("\n1. 数据加载与探索")
print("-"*50)

# 读取数据
data = pd.read_excel('policy_data.xlsx')
print(f"数据集大小: {data.shape}")

# 查看目标变量分布
print("\n目标变量(renewal)分布:")
renewal_counts = data['renewal'].value_counts()
print(renewal_counts)
print(f"续保率: {renewal_counts['Yes'] / len(data) * 100:.2f}%")

# 2. 数据预处理
print("\n2. 数据预处理")
print("-"*50)

# 检查缺失值
print("\n检查缺失值:")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.any() > 0 else "无缺失值")

# 将目标变量转换为二进制
data['renewal_binary'] = (data['renewal'] == 'Yes').astype(int)
print("\n目标变量转换为二进制:")
print(data['renewal_binary'].value_counts())

# 特征选择
# 排除不需要的列：policy_id(ID列), renewal(原始目标变量), policy_start_date, policy_end_date(日期列)
features = data.drop(['policy_id', 'renewal', 'renewal_binary', 'policy_start_date', 'policy_end_date'], axis=1)

# 识别数值型和分类型特征
numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = features.select_dtypes(include=['object']).columns.tolist()

print("\n数值型特征:")
print(numeric_features)
print("\n分类型特征:")
print(categorical_features)

# 3. 特征工程
print("\n3. 特征工程")
print("-"*50)

# 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. 模型构建
print("\n4. 模型构建")
print("-"*50)

# 准备数据
X = features
y = data['renewal_binary']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 创建逻辑回归模型管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# 5. 模型训练与评估
print("\n5. 模型训练与评估")
print("-"*50)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上评估
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['不续保', '续保']))

# 混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 计算ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"\nAUC: {roc_auc:.4f}")

# 6. 模型优化
print("\n6. 模型优化")
print("-"*50)

# 使用网格搜索优化超参数
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs'],
    'classifier__penalty': ['l1', 'l2']
}

# 注意：liblinear求解器支持l1和l2正则化，而lbfgs只支持l2正则化
# 创建一个自定义参数列表，避免不兼容的组合
param_list = [
    {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__solver': ['liblinear'], 'classifier__penalty': ['l1', 'l2']},
    {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__solver': ['lbfgs'], 'classifier__penalty': ['l2']}
]

grid_search = GridSearchCV(model, param_list, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

# 评估最佳模型
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"\n最佳模型准确率: {accuracy_best:.4f}")

print("\n最佳模型分类报告:")
print(classification_report(y_test, y_pred_best, target_names=['不续保', '续保']))

# 计算最佳模型的ROC曲线
fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_proba_best)
roc_auc_best = auc(fpr_best, tpr_best)
print(f"\n最佳模型AUC: {roc_auc_best:.4f}")

# 7. 特征重要性分析
print("\n7. 特征重要性分析")
print("-"*50)

# 获取逻辑回归系数
# 由于使用了OneHotEncoder，我们需要获取特征名称
preprocessor = best_model.named_steps['preprocessor']
classifier = best_model.named_steps['classifier']

# 获取数值特征名称
numeric_features_transformed = numeric_features

# 获取分类特征转换后的名称
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
categorical_features_transformed = ohe.get_feature_names_out(categorical_features)

# 合并所有特征名称
feature_names = list(numeric_features_transformed) + list(categorical_features_transformed)

# 获取系数
coefficients = classifier.coef_[0]

# 创建特征重要性DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# 按系数绝对值排序
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\n特征重要性(前20):")
print(feature_importance.head(20))

# 8. 可视化
print("\n8. 可视化")
print("-"*50)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'基础模型 (AUC = {roc_auc:.4f})')
plt.plot(fpr_best, tpr_best, label=f'优化模型 (AUC = {roc_auc_best:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

# 绘制特征重要性
plt.figure(figsize=(12, 10))
top_features = feature_importance.head(15)
sns.barplot(x='Coefficient', y='Feature', data=top_features)
plt.title('逻辑回归模型 - 特征重要性(前15)')
plt.xlabel('系数')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues',
            xticklabels=['不续保', '续保'], yticklabels=['不续保', '续保'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("\n分析完成！结果已保存到当前目录。")
print("生成的图表文件:")
print("- roc_curve.png: ROC曲线")
print("- feature_importance.png: 特征重要性")
print("- confusion_matrix.png: 混淆矩阵")

# 9. 模型保存
print("\n9. 模型保存")
print("-"*50)

import joblib
joblib.dump(best_model, 'logistic_regression_model.pkl')
print("模型已保存为 'logistic_regression_model.pkl'")

# 10. 总结
print("\n10. 模型总结")
print("-"*50)
print(f"最终模型准确率: {accuracy_best:.4f}")
print(f"最终模型AUC: {roc_auc_best:.4f}")
print("\n最重要的5个特征:")
for i, row in feature_importance.head(5).iterrows():
    print(f"- {row['Feature']}: {row['Coefficient']:.4f}")

print("\n模型解释:")
print("正系数表示该特征增加续保概率，负系数表示该特征降低续保概率。")
print("系数的绝对值大小表示该特征对预测结果的影响程度。")

# 关闭输出文件
output_file.close()
sys.stdout = sys.__stdout__
print("逻辑回归模型分析完成！结果已保存到 logistic_regression_results.txt 文件中。")