import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV文件
try:
    df = pd.read_csv('train.csv')
    print(f"成功读取数据文件，共 {df.shape[0]} 行，{df.shape[1]} 列")
    
    # 显示前几行数据
    print("\n数据预览:")
    print(df.head())
    
    # 显示列名和数据类型
    print("\n列名和数据类型:")
    print(df.dtypes)
    
    # 基本统计信息
    print("\n数值型数据统计信息:")
    print(df.describe())
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值统计:")
    print(missing_values[missing_values > 0])
    
    # 分析分类变量
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    print("\n分类变量:")
    for col in categorical_columns:
        print(f"\n{col} 的唯一值数量: {df[col].nunique()}")
        print(df[col].value_counts().head())
    
    # 数据可视化
    print("\n开始生成数据可视化...")
    
    # 设置图表风格
    sns.set(style="whitegrid")
    
    # 1. 数值型数据分布
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(15, len(numeric_columns) * 3))
        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(len(numeric_columns), 2, i*2-1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'{col} 分布')
            
            plt.subplot(len(numeric_columns), 2, i*2)
            sns.boxplot(x=df[col].dropna())
            plt.title(f'{col} 箱线图')
        plt.tight_layout()
        plt.savefig('numeric_distributions.png')
        print("数值型数据分布图已保存为 numeric_distributions.png")
    
    # 2. 分类数据分布
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            if df[col].nunique() < 30:  # 只对唯一值较少的分类变量绘图
                plt.figure(figsize=(10, 6))
                df[col].value_counts().plot(kind='bar')
                plt.title(f'{col} 分布')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{col}_distribution.png')
                print(f"{col} 分布图已保存为 {col}_distribution.png")
    
    # 3. 相关性分析
    if len(numeric_columns) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_columns].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        print("相关性热力图已保存为 correlation_heatmap.png")
    
    print("\n数据分析完成！")
    
except FileNotFoundError:
    print("错误：找不到 train.csv 文件，请确保文件路径正确")
except Exception as e:
    print(f"发生错误：{str(e)}")