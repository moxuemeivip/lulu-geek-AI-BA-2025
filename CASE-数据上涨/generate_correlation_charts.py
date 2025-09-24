import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.gridspec import GridSpec

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建保存图表的文件夹
output_folder = 'CorrelationCharts'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已创建文件夹: {output_folder}")

# 读取Excel文件
try:
    # 读取数据
    df = pd.read_excel('不同维度的数据统计.xlsx')
    
    print("成功读取文件并开始绘制相关性图表...\n")
    
    # 转换日期列
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 提取需要分析的列
    analysis_df = df[['访问量', '新用户', '老用户']].copy()
    
    # 计算相关性矩阵
    corr_matrix = analysis_df.corr(method='pearson')
    print("相关性矩阵:")
    print(corr_matrix)
    print("\n")
    
    # 1. 绘制散点图矩阵
    print("绘制散点图矩阵...")
    plt.figure(figsize=(12, 10))
    # 创建自定义网格
    gs = GridSpec(3, 3)
    
    # 访问量与新用户的散点图
    ax1 = plt.subplot(gs[0, 1])
    sns.scatterplot(x='访问量', y='新用户', data=df, ax=ax1, color='blue', s=100)
    ax1.set_title('访问量与新用户的关系', fontsize=14)
    # 添加趋势线
    z = np.polyfit(df['访问量'], df['新用户'], 1)
    p = np.poly1d(z)
    ax1.plot(df['访问量'], p(df['访问量']), "r--")
    # 显示相关系数
    corr_visit_new = corr_matrix.loc['访问量', '新用户']
    ax1.text(0.05, 0.95, f'相关系数: {corr_visit_new:.4f}', 
             transform=ax1.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 访问量与老用户的散点图
    ax2 = plt.subplot(gs[0, 2])
    sns.scatterplot(x='访问量', y='老用户', data=df, ax=ax2, color='green', s=100)
    ax2.set_title('访问量与老用户的关系', fontsize=14)
    # 添加趋势线
    z = np.polyfit(df['访问量'], df['老用户'], 1)
    p = np.poly1d(z)
    ax2.plot(df['访问量'], p(df['访问量']), "r--")
    # 显示相关系数
    corr_visit_old = corr_matrix.loc['访问量', '老用户']
    ax2.text(0.05, 0.95, f'相关系数: {corr_visit_old:.4f}', 
             transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 新用户与老用户的散点图
    ax3 = plt.subplot(gs[1, 2])
    sns.scatterplot(x='新用户', y='老用户', data=df, ax=ax3, color='purple', s=100)
    ax3.set_title('新用户与老用户的关系', fontsize=14)
    # 添加趋势线
    z = np.polyfit(df['新用户'], df['老用户'], 1)
    p = np.poly1d(z)
    ax3.plot(df['新用户'], p(df['新用户']), "r--")
    # 显示相关系数
    corr_new_old = corr_matrix.loc['新用户', '老用户']
    ax3.text(0.05, 0.95, f'相关系数: {corr_new_old:.4f}', 
             transform=ax3.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 添加直方图对角线
    ax4 = plt.subplot(gs[1, 1])
    sns.histplot(df['访问量'], ax=ax4, kde=True, color='skyblue')
    ax4.set_title('访问量分布', fontsize=14)
    
    ax5 = plt.subplot(gs[2, 2])
    sns.histplot(df['老用户'], ax=ax5, kde=True, color='lightgreen')
    ax5.set_title('老用户分布', fontsize=14)
    
    ax6 = plt.subplot(gs[2, 1])
    sns.histplot(df['新用户'], ax=ax6, kde=True, color='lightcoral')
    ax6.set_title('新用户分布', fontsize=14)
    
    # 隐藏空白子图
    plt.delaxes(plt.subplot(gs[1, 0]))
    plt.delaxes(plt.subplot(gs[2, 0]))
    plt.delaxes(plt.subplot(gs[0, 0]))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '相关性散点图矩阵.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"散点图矩阵已保存到: {os.path.join(output_folder, '相关性散点图矩阵.png')}")
    
    # 2. 绘制热力图
    print("绘制相关性热力图...")
    plt.figure(figsize=(10, 8))
    # 创建热力图
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                          square=True, linewidths=.5, vmin=-1, vmax=1,
                          annot_kws={'size': 14, 'weight': 'bold'})
    # 自定义标题
    plt.title('访问量、新用户和老用户相关性热力图', fontsize=16, pad=20)
    # 调整坐标轴标签
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    # 保存图表
    plt.savefig(os.path.join(output_folder, '相关性热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"相关性热力图已保存到: {os.path.join(output_folder, '相关性热力图.png')}")
    
    # 3. 绘制雷达图
    print("绘制雷达图...")
    # 准备雷达图数据
    # 计算各个指标的标准化值 (0-100)
    normalized_data = {
        '访问量': [(x - df['访问量'].min()) / (df['访问量'].max() - df['访问量'].min()) * 100 
                  for x in df['访问量']],
        '新用户': [(x - df['新用户'].min()) / (df['新用户'].max() - df['新用户'].min()) * 100 
                  for x in df['新用户']],
        '老用户': [(x - df['老用户'].min()) / (df['老用户'].max() - df['老用户'].min()) * 100 
                  for x in df['老用户']]
    }
    
    # 创建DataFrame
    radar_df = pd.DataFrame(normalized_data, index=df['日期'].dt.strftime('%Y-%m-%d'))
    
    # 设置雷达图参数
    categories = list(radar_df.columns)
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # 创建雷达图
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    
    # 绘制每条数据线
    for i, date in enumerate(radar_df.index):
        values = radar_df.iloc[i].tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=date)
        ax.fill(angles, values, alpha=0.1)
    
    # 设置角度和标签
    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.set_ylim(0, 100)
    plt.yticks([20, 40, 60, 80, 100], ['20', '40', '60', '80', '100'], fontsize=10)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), title="日期", fontsize=10)
    
    # 添加标题
    plt.title('访问量、新用户和老用户雷达图分析\n(标准化值)', fontsize=16, pad=20)
    
    # 保存图表
    plt.savefig(os.path.join(output_folder, '访问量与用户雷达图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"雷达图已保存到: {os.path.join(output_folder, '访问量与用户雷达图.png')}")
    
    # 4. 绘制时间序列趋势图
    print("绘制时间序列趋势图...")
    plt.figure(figsize=(14, 8))
    
    # 创建双Y轴图表
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    
    # 绘制访问量和新用户（共享主Y轴）
    line1 = ax1.plot(df['日期'], df['访问量'], 'b-', linewidth=3, marker='o', markersize=8, label='访问量')
    line2 = ax1.plot(df['日期'], df['新用户'], 'r-', linewidth=2, marker='s', markersize=8, label='新用户')
    
    # 绘制老用户（使用次Y轴）
    line3 = ax2.plot(df['日期'], df['老用户'], 'g-', linewidth=2, marker='^', markersize=8, label='老用户')
    
    # 设置标签和标题
    ax1.set_xlabel('日期', fontsize=14)
    ax1.set_ylabel('访问量/新用户', fontsize=14)
    ax2.set_ylabel('老用户', fontsize=14)
    plt.title('访问量、新用户和老用户时间序列趋势', fontsize=16)
    
    # 设置图例
    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12)
    
    # 格式化日期显示
    plt.xticks(rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_folder, '时间序列趋势图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"时间序列趋势图已保存到: {os.path.join(output_folder, '时间序列趋势图.png')}")
    
    print("\n所有图表生成完成!")
    print(f"共生成了4个图表文件，保存在 {output_folder} 文件夹中")
    
except Exception as e:
    print(f"绘制图表过程中发生错误: {e}")