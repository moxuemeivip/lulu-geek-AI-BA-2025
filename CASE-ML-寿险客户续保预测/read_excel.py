import pandas as pd
import numpy as np

# 读取Excel文件
data = pd.read_excel('policy_data.xlsx')

# 显示数据表的基本信息
print("数据表基本信息:")
print(f"行数: {data.shape[0]}, 列数: {data.shape[1]}")
print("\n数据表字段:")
for col in data.columns:
    print(f"- {col}")

# 显示数据类型
print("\n数据类型:")
print(data.dtypes)

# 显示前5行数据
print("\n前5行数据:")
print(data.head(5).to_string())

# 显示字段的基本统计信息
print("\n数值型字段的基本统计信息:")
print(data.describe().to_string())