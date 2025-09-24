import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("开始加载数据...")
# 读取数据
df = pd.read_csv('train.csv')

# 数据基本信息
print(f"数据集形状: {df.shape}")
print("\n数据集前5行:")
print(df.head())

# 检查目标变量分布
print("\n目标变量分布:")
print(df['subscribe'].value_counts())
print(f"认购率: {df['subscribe'].value_counts(normalize=True)['yes']*100:.2f}%")

# 数据预处理
print("\n开始数据预处理...")

# 处理分类变量
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 将目标变量转换为数值
df['subscribe'] = df['subscribe'].map({'yes': 1, 'no': 0})

# 删除不需要的列
df = df.drop(['id'], axis=1)

# 特征和目标变量分离
X = df.drop('subscribe', axis=1)
y = df['subscribe']

# 标准化特征 - 对逻辑回归很重要
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 构建随机森林模型
print("\n开始训练随机森林模型...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 模型评估
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n随机森林模型评估结果:")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n随机森林分类报告:")
print(classification_report(y_test, y_pred))

# 构建逻辑回归模型
print("\n开始训练逻辑回归模型...")
# 使用L2正则化，C值较小以增加正则化强度
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# 逻辑回归模型评估
lr_y_pred = lr_model.predict(X_test)
lr_y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

print("\n逻辑回归模型评估结果:")
print(f"准确率: {accuracy_score(y_test, lr_y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, lr_y_pred_proba):.4f}")

print("\n逻辑回归分类报告:")
print(classification_report(y_test, lr_y_pred))

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('随机森林混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')

# 逻辑回归混淆矩阵
plt.figure(figsize=(8, 6))
lr_cm = confusion_matrix(y_test, lr_y_pred)
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('逻辑回归混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('lr_confusion_matrix.png', dpi=300, bbox_inches='tight')

# 随机森林特征重要性分析
print("\n随机森林特征重要性分析...")
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('重要性', ascending=False)

print("\n随机森林前10个最重要的特征:")
print(feature_importance.head(10))

# 保存完整的特征重要性排名到CSV文件
feature_importance.to_csv('rf_feature_importance_ranking.csv', index=False, encoding='utf-8-sig')
print("\n随机森林特征重要性排名已保存到 rf_feature_importance_ranking.csv")

# 逻辑回归系数分析
print("\n逻辑回归系数分析...")
# 创建包含特征名称和系数的DataFrame
lr_coefficients = pd.DataFrame({
    '特征': X.columns,
    '系数': lr_model.coef_[0],
    '系数绝对值': np.abs(lr_model.coef_[0])
})

# 按系数绝对值从大到小排序
lr_coefficients = lr_coefficients.sort_values('系数绝对值', ascending=False)

print("\n逻辑回归前10个最重要的特征及其系数:")
print(lr_coefficients.head(10))

# 保存完整的逻辑回归系数到CSV文件
lr_coefficients.to_csv('lr_coefficients_ranking.csv', index=False, encoding='utf-8-sig')
print("\n逻辑回归系数排名已保存到 lr_coefficients_ranking.csv")

# 逻辑回归系数可视化 - 按绝对值排序但保留正负号
fig, ax = plt.subplots(figsize=(14, 12))
# 获取前15个特征（按绝对值排序）
top_coefficients = lr_coefficients.head(15).copy()

# 创建颜色映射：正系数为蓝色，负系数为红色
colors = ['red' if c < 0 else 'blue' for c in top_coefficients['系数']]

# 创建柱状图
bars = ax.barh(top_coefficients['特征'], top_coefficients['系数'], color=colors)

# 添加垂直线表示零点
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# 在柱状图上添加数值标签
for i, v in enumerate(top_coefficients['系数']):
    if v < 0:
        # 负值标签放在左侧
        ax.text(v - 0.05, i, f"{v:.4f}", va='center', ha='right')
    else:
        # 正值标签放在右侧
        ax.text(v + 0.05, i, f"{v:.4f}", va='center', ha='left')

plt.title('逻辑回归系数 (按绝对值排序，保留正负号)', fontsize=16, fontweight='bold')
plt.xlabel('系数值', fontsize=14)
plt.ylabel('特征', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 添加图例说明
plt.text(0.95, 0.05, '蓝色 = 正向影响 (增加订阅概率)\n红色 = 负向影响 (降低订阅概率)', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig('lr_coefficients.png', dpi=300, bbox_inches='tight')

# 创建另一个版本的逻辑回归系数可视化 - 显示所有特征
fig, ax = plt.subplots(figsize=(16, 14))
# 获取所有特征
all_coefficients = lr_coefficients.copy()

# 创建颜色映射：正系数为蓝色，负系数为红色
colors = ['red' if c < 0 else 'blue' for c in all_coefficients['系数']]

# 创建柱状图
bars = ax.barh(all_coefficients['特征'], all_coefficients['系数'], color=colors)

# 添加垂直线表示零点
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# 在柱状图上添加数值标签
for i, v in enumerate(all_coefficients['系数']):
    if v < 0:
        # 负值标签放在左侧
        ax.text(v - 0.05, i, f"{v:.4f}", va='center', ha='right')
    else:
        # 正值标签放在右侧
        ax.text(v + 0.05, i, f"{v:.4f}", va='center', ha='left')

plt.title('所有逻辑回归系数 (按绝对值排序，保留正负号)', fontsize=16, fontweight='bold')
plt.xlabel('系数值', fontsize=14)
plt.ylabel('特征', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 添加图例说明
plt.text(0.95, 0.05, '蓝色 = 正向影响 (增加订阅概率)\n红色 = 负向影响 (降低订阅概率)', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig('lr_all_coefficients.png', dpi=300, bbox_inches='tight')

# 随机森林特征重要性可视化
fig, ax = plt.subplots(figsize=(14, 10))
# 使用更鲜明的颜色映射
colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_importance.head(15))))
sns.barplot(x='重要性', y='特征', data=feature_importance.head(15), palette=colors, ax=ax)

# 在柱状图上添加数值标签
for i, v in enumerate(feature_importance.head(15)['重要性']):
    ax.text(v + 0.01, i, f"{v:.4f}", va='center')

plt.title('随机森林特征重要性排序 (Top 15)', fontsize=16, fontweight='bold')
plt.xlabel('重要性', fontsize=14)
plt.ylabel('特征', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')

# ROC曲线比较
fig, ax = plt.subplots(figsize=(10, 8))
# 随机森林ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba)
ax.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})', linewidth=2)

# 逻辑回归ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_y_pred_proba)
ax.plot(fpr_lr, tpr_lr, label=f'逻辑回归 (AUC = {roc_auc_score(y_test, lr_y_pred_proba):.4f})', linewidth=2, linestyle='--')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('假正例率 (FPR)', fontsize=12)
ax.set_ylabel('真正例率 (TPR)', fontsize=12)
ax.set_title('ROC曲线比较', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')

# 分析前5个重要特征的分布
top_features = feature_importance['特征'].head(5).tolist()
print(f"\n分析前5个随机森林重要特征: {top_features}")

# 创建一个图表，展示前5个重要特征与目标变量的关系
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    if i < 5:  # 只取前5个特征
        if df[feature].dtype == 'object' or df[feature].nunique() < 10:
            # 分类特征
            sns.countplot(x=feature, hue='subscribe', data=df, ax=axes[i])
        else:
            # 数值特征
            sns.boxplot(x='subscribe', y=feature, data=df, ax=axes[i])
        
        axes[i].set_title(f'特征: {feature}', fontsize=14, fontweight='bold')
        axes[i].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('top_features_analysis.png', dpi=300, bbox_inches='tight')

print("\n分析完成，所有图表已保存。") 