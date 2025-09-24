import pandas as pd
import sys
import numpy as np
# 设置显示选项以确保完整显示所有内容
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 2000)  # 增加显示宽度
pd.set_option('display.max_colwidth', 100)  # 设置列宽
pd.set_option('display.max_rows', 20)  # 显示更多行
pd.set_option('display.precision', 3)  # 设置显示精度
# 设置输出编码，确保中文正常显示
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

# 读取Excel文件
excel_file_path = '不同维度的数据统计.xlsx'
try:
    # 使用pandas读取Excel文件
    df = pd.read_excel(excel_file_path)
    
    print(f"成功读取文件: {excel_file_path}")
    print(f"数据表的形状: {df.shape}")  # 显示行数和列数
    
    # 转换日期列的显示格式，避免时区问题
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
    
    # 显示数值列的基本统计信息
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("\n数值列的基本统计信息:\n")
        print(df[numeric_cols].describe())
        
        # 计算并显示相关性矩阵
        print("\n数值型数据维度之间的相关性矩阵 (Pearson相关系数):\n")
        corr_matrix = df[numeric_cols].corr(method='pearson')
        
        # 打印相关性矩阵，确保格式清晰
        print("\n相关性矩阵:\n")
        print(corr_matrix)
        
        # 显示高度相关的维度对（相关系数绝对值大于0.7）
        print("\n高度相关的维度对 (相关系数绝对值 > 0.7):\n")
        corr_pairs = []
        # 遍历相关性矩阵，找出高度相关的维度对
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
        
        if corr_pairs:
            # 按相关系数绝对值大小排序
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            # 打印结果
            for col1, col2, corr in corr_pairs:
                print(f"{col1} 与 {col2}: {corr:.4f}")
        else:
            print("没有高度相关的维度对。")
    else:
        print("\n数据表中没有数值类型的列。")
    
except Exception as e:
    print(f"读取文件时发生错误: {e}")