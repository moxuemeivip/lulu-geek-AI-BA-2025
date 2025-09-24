import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys

# 将输出重定向到文件
output_file = open('analysis_results.txt', 'w', encoding='utf-8')
sys.stdout = output_file

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取Excel文件
data = pd.read_excel('policy_data.xlsx')

# 1. 年龄分布分析
print("="*50)
print("1. 年龄分布分析")
print("="*50)

# 基本统计量
age_stats = data['age'].describe()
print("年龄基本统计量:")
print(age_stats)

# 年龄分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=20, kde=True)
plt.title('客户年龄分布')
plt.xlabel('年龄')
plt.ylabel('频数')
plt.savefig('age_distribution.png')
plt.close()

# 年龄段分布
age_bins = [0, 20, 30, 40, 50, 60, 70]
age_labels = ['20岁以下', '20-30岁', '30-40岁', '40-50岁', '50-60岁', '60岁以上']
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)
age_group_counts = data['age_group'].value_counts().sort_index()
print("\n年龄段分布:")
print(age_group_counts)

# 不同年龄段的续保率
renewal_by_age = data.groupby('age_group')['renewal'].apply(lambda x: (x == 'Yes').mean() * 100)
print("\n不同年龄段的续保率(%):")
print(renewal_by_age)

# 2. 性别差异分析
print("\n" + "="*50)
print("2. 性别差异分析")
print("="*50)

# 性别分布
gender_counts = data['gender'].value_counts()
print("性别分布:")
print(gender_counts)
print(f"男性比例: {gender_counts['男'] / len(data) * 100:.2f}%")
print(f"女性比例: {gender_counts['女'] / len(data) * 100:.2f}%")

# 不同性别的续保率
renewal_by_gender = data.groupby('gender')['renewal'].apply(lambda x: (x == 'Yes').mean() * 100)
print("\n不同性别的续保率(%):")
print(renewal_by_gender)

# 不同性别的年龄分布
print("\n不同性别的年龄统计:")
gender_age_stats = data.groupby('gender')['age'].describe()
print(gender_age_stats)

# 不同性别的保费金额
print("\n不同性别的保费金额统计:")
gender_premium_stats = data.groupby('gender')['premium_amount'].describe()
print(gender_premium_stats)

# 3. 出生地区与投保所在地区的关联分析
print("\n" + "="*50)
print("3. 出生地区与投保所在地区的关联分析")
print("="*50)

# 出生地区分布
birth_region_counts = data['birth_region'].value_counts().head(10)
print("出生地区分布(前10):")
print(birth_region_counts)

# 投保地区分布
insurance_region_counts = data['insurance_region'].value_counts().head(10)
print("\n投保地区分布(前10):")
print(insurance_region_counts)

# 出生地区与投保地区一致的比例
same_region = (data['birth_region'] == data['insurance_region']).mean() * 100
print(f"\n出生地区与投保地区一致的比例: {same_region:.2f}%")

# 创建出生地区与投保地区的关联表
region_relation = pd.crosstab(data['birth_region'], data['insurance_region'])
print("\n出生地区与投保地区的关联表(部分展示):")
print(region_relation.iloc[:5, :5])  # 只显示部分数据，避免输出过多

# 出生地区与投保地区一致性对续保的影响
data['same_region'] = (data['birth_region'] == data['insurance_region']).astype(int)
renewal_by_same_region = data.groupby('same_region')['renewal'].apply(lambda x: (x == 'Yes').mean() * 100)
print("\n出生地区与投保地区一致性对续保的影响(%):")
print("0=不一致, 1=一致")
print(renewal_by_same_region)

# 关闭输出文件
output_file.close()
sys.stdout = sys.__stdout__
print("分析结果已保存到 analysis_results.txt 文件中。")