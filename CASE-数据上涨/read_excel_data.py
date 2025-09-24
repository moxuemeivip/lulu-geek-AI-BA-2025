import pandas as pd

# 读取Excel文件
excel_file_path = '是否有活动.xlsx'
try:
    # 使用pandas读取Excel文件
    df = pd.read_excel(excel_file_path)
    
    print(f"成功读取文件: {excel_file_path}")
    print(f"数据表的形状: {df.shape}")  # 显示行数和列数
    print("\n数据表的前5行:\n")
    print(df.head())  # 显示前5行数据
    
    print("\n数据表的列名:\n")
    print(df.columns.tolist())  # 显示所有列名
    
    print("\n数据表的基本信息:\n")
    print(df.info())  # 显示数据表的基本信息
    
    # 检查是否有缺失值
    print("\n缺失值统计:\n")
    print(df.isnull().sum())  # 统计每列的缺失值数量
    
    # 计算访问量与推广活动的Pearson相关系数
    print("\n计算访问量与推广活动的Pearson相关系数:\n")
    
    # 将推广活动列转换为数值型（"有"=1，"无"=0）
    df['推广活动数值'] = df['推广活动'].map({'有': 1, '无': 0})
    
    # 计算Pearson相关系数
    pearson_corr = df['访问量'].corr(df['推广活动数值'], method='pearson')
    print(f"访问量与推广活动的Pearson相关系数: {pearson_corr:.4f}")
    
    # 显示转换后的数据集前5行
    print("\n转换后的数据集前5行:\n")
    print(df[['日期', '访问量', '推广活动', '推广活动数值']].head())
    
except Exception as e:
    print(f"读取文件时发生错误: {e}")