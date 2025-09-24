import pandas as pd
import numpy as np

# 读取Excel文件
try:
    # 读取数据
    df = pd.read_excel('不同维度的数据统计.xlsx')
    
    print("成功读取文件并开始分析访问量与新用户、老用户之间的关系...\n")
    
    # 转换日期列的显示格式
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
    
    # 提取需要分析的列
    analysis_df = df[['日期', '访问量', '新用户', '老用户']].copy()
    
    # 显示数据的前10行
    print("数据预览 (前10行):")
    print(analysis_df)
    print("\n")
    
    # 计算基本统计信息
    print("基本统计信息:")
    print(analysis_df.describe())
    print("\n")
    
    # 计算相关性系数
    print("相关性分析:")
    corr_matrix = analysis_df[['访问量', '新用户', '老用户']].corr(method='pearson')
    print("Pearson相关系数矩阵:")
    print(corr_matrix)
    print("\n")
    
    # 计算新用户和老用户占比
    analysis_df['新用户占比'] = analysis_df['新用户'] / analysis_df['访问量']
    analysis_df['老用户占比'] = analysis_df['老用户'] / analysis_df['访问量']
    
    print("新用户和老用户占比统计:")
    print(analysis_df[['日期', '新用户占比', '老用户占比']])
    print("\n")
    
    print("占比汇总统计:")
    print(f"平均新用户占比: {analysis_df['新用户占比'].mean():.2%}")
    print(f"平均老用户占比: {analysis_df['老用户占比'].mean():.2%}")
    print(f"新用户占比标准差: {analysis_df['新用户占比'].std():.4f}")
    print(f"老用户占比标准差: {analysis_df['老用户占比'].std():.4f}")
    print("\n")
    
    # 进行简单的线性回归分析（使用numpy）
    print("线性回归分析 (访问量 ~ 新用户 + 老用户):")
    # 准备数据
    X = analysis_df[['新用户', '老用户']].values
    y = analysis_df['访问量'].values
    
    # 添加截距项
    X = np.column_stack((np.ones(len(X)), X))
    
    # 计算回归系数
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    print(f"截距: {beta[0]:.4f}")
    print(f"新用户系数: {beta[1]:.4f}")
    print(f"老用户系数: {beta[2]:.4f}")
    
    # 计算R²
    y_pred = X.dot(beta)
    ss_total = np.sum((y - np.mean(y))** 2)
    ss_residual = np.sum((y - y_pred)** 2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f"R² 决定系数: {r_squared:.4f}")
    print("\n")
    
    # 观察日期趋势
    print("日期趋势分析:")
    print("访问量、新用户和老用户随时间的变化:")
    # 格式化输出，确保对齐
    print("日期       访问量     新用户     老用户     新用户占比  老用户占比")
    print("-------------------------------------------------------")
    for _, row in analysis_df.iterrows():
        print(f"{row['日期']}  {row['访问量']:8d}  {row['新用户']:8d}  {row['老用户']:8d}  {row['新用户占比']:6.1%}  {row['老用户占比']:6.1%}")
    
    print("\n\n分析总结:")
    print("1. 相关性分析显示，访问量与新用户高度相关(相关系数: {:.4f})，与老用户也有较强相关性(相关系数: {:.4f})" 
          .format(corr_matrix.loc['访问量', '新用户'], corr_matrix.loc['访问量', '老用户']))
    print("2. 平均而言，新用户占总访问量的比例为 {:.1%}，老用户占比为 {:.1%}".format(analysis_df['新用户占比'].mean(), analysis_df['老用户占比'].mean()))
    print("3. 线性回归模型显示，新用户和老用户能够很好地解释访问量的变化(R² = {:.4f})" .format(r_squared))
    print("4. 从趋势上看，随着访问量的增加，新用户和老用户的数量也呈现增长趋势。")
    
except Exception as e:
    print(f"分析过程中发生错误: {e}")