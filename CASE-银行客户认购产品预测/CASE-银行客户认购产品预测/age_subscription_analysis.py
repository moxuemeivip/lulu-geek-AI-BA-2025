import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_csv('train.csv')

# 查看年龄的基本统计信息
print("年龄的基本统计信息:")
print(df['age'].describe())

# 定义年龄段
age_bins = [0, 20, 30, 40, 50, 60, 70, 100]
age_labels = ['20岁以下', '21-30岁', '31-40岁', '41-50岁', '51-60岁', '61-70岁', '70岁以上']

# 添加年龄段列
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# 计算每个年龄段的认购比例
subscription_by_age = df.groupby('age_group')['subscribe'].apply(lambda x: (x == 'yes').mean()).reset_index()
subscription_by_age.columns = ['年龄段', '认购比例']
subscription_by_age['认购比例'] = subscription_by_age['认购比例'] * 100  # 转换为百分比

# 打印结果
print("\n不同年龄段的认购比例:")
print(subscription_by_age)

# 计算每个年龄段的样本数量
age_counts = df['age_group'].value_counts().sort_index().reset_index()
age_counts.columns = ['年龄段', '样本数量']

# 合并认购比例和样本数量
result = pd.merge(subscription_by_age, age_counts, on='年龄段')
print("\n不同年龄段的认购比例和样本数量:")
print(result)

# 可视化不同年龄段的认购比例
plt.figure(figsize=(12, 6))

# 创建柱状图
ax = sns.barplot(x='年龄段', y='认购比例', data=subscription_by_age, palette='viridis')

# 在柱状图上添加数值标签
for i, row in subscription_by_age.iterrows():
    ax.text(i, row['认购比例'] + 0.5, f"{row['认购比例']:.2f}%", ha='center')

# 添加样本数量标签
for i, row in result.iterrows():
    ax.text(i, 0.5, f"n={row['样本数量']}", ha='center', color='white', fontweight='bold')

plt.title('不同年龄段的认购比例', fontsize=15)
plt.xlabel('年龄段', fontsize=12)
plt.ylabel('认购比例 (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('age_subscription_ratio.png', dpi=300)

# 创建年龄分布直方图
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='subscribe', bins=20, multiple='stack')
plt.title('不同年龄的认购分布', fontsize=15)
plt.xlabel('年龄', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('age_subscription_distribution.png', dpi=300)

# 计算每个年龄的认购比例
age_subscription = df.groupby('age')['subscribe'].apply(lambda x: (x == 'yes').mean() * 100).reset_index()
age_subscription.columns = ['年龄', '认购比例']

# 创建年龄与认购比例的折线图
plt.figure(figsize=(14, 6))
sns.lineplot(x='年龄', y='认购比例', data=age_subscription, marker='o')
plt.title('各年龄的认购比例', fontsize=15)
plt.xlabel('年龄', fontsize=12)
plt.ylabel('认购比例 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('age_subscription_line.png', dpi=300)

print("\n分析完成，图表已保存。") 