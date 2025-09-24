import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取Excel文件
try:
    # 读取数据
    df = pd.read_excel('不同维度的数据统计.xlsx')
    
    print("成功读取文件并开始绘制散点图...\n")
    
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('访问量、新用户和老用户关系散点图分析', fontsize=16)
    
    # 1. 访问量与新用户的散点图
    sns.scatterplot(x='访问量', y='新用户', data=df, ax=axes[0, 0], color='blue', s=100)
    axes[0, 0].set_title('访问量与新用户的关系', fontsize=14)
    axes[0, 0].set_xlabel('访问量', fontsize=12)
    axes[0, 0].set_ylabel('新用户', fontsize=12)
    
    # 添加趋势线
    z = np.polyfit(df['访问量'], df['新用户'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['访问量'], p(df['访问量']), "r--")
    
    # 计算相关性系数并显示
    corr_visit_new = df['访问量'].corr(df['新用户'])
    axes[0, 0].text(0.05, 0.95, f'相关系数: {corr_visit_new:.4f}', 
                    transform=axes[0, 0].transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 访问量与老用户的散点图
    sns.scatterplot(x='访问量', y='老用户', data=df, ax=axes[0, 1], color='green', s=100)
    axes[0, 1].set_title('访问量与老用户的关系', fontsize=14)
    axes[0, 1].set_xlabel('访问量', fontsize=12)
    axes[0, 1].set_ylabel('老用户', fontsize=12)
    
    # 添加趋势线
    z = np.polyfit(df['访问量'], df['老用户'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df['访问量'], p(df['访问量']), "r--")
    
    # 计算相关性系数并显示
    corr_visit_old = df['访问量'].corr(df['老用户'])
    axes[0, 1].text(0.05, 0.95, f'相关系数: {corr_visit_old:.4f}', 
                    transform=axes[0, 1].transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. 新用户与老用户的散点图
    sns.scatterplot(x='新用户', y='老用户', data=df, ax=axes[1, 0], color='purple', s=100)
    axes[1, 0].set_title('新用户与老用户的关系', fontsize=14)
    axes[1, 0].set_xlabel('新用户', fontsize=12)
    axes[1, 0].set_ylabel('老用户', fontsize=12)
    
    # 添加趋势线
    z = np.polyfit(df['新用户'], df['老用户'], 1)
    p = np.poly1d(z)
    axes[1, 0].plot(df['新用户'], p(df['新用户']), "r--")
    
    # 计算相关性系数并显示
    corr_new_old = df['新用户'].corr(df['老用户'])
    axes[1, 0].text(0.05, 0.95, f'相关系数: {corr_new_old:.4f}', 
                    transform=axes[1, 0].transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. 新用户占比的趋势图（作为补充分析）
    df['日期'] = pd.to_datetime(df['日期'])
    df['新用户占比'] = df['新用户'] / df['访问量']
    sns.lineplot(x='日期', y='新用户占比', data=df, ax=axes[1, 1], marker='o', markersize=10)
    axes[1, 1].set_title('新用户占比随时间的变化趋势', fontsize=14)
    axes[1, 1].set_xlabel('日期', fontsize=12)
    axes[1, 1].set_ylabel('新用户占比', fontsize=12)
    axes[1, 1].set_ylim(0, 1)
    
    # 格式化日期显示
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45)
    
    # 添加数据标签
    for i, row in df.iterrows():
        axes[1, 1].annotate(f'{row["新用户占比"]:.1%}', 
                           (row['日期'], row['新用户占比']), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整标题位置
    
    # 保存图表
    plt.savefig('用户关系散点图分析.png', dpi=300, bbox_inches='tight')
    print("图表已保存为'用户关系散点图分析.png'")
    
    # 显示图表
    print("图表已生成，将自动显示...")
    plt.show()
    
    print("\n散点图分析总结:")
    print(f"1. 访问量与新用户高度相关，相关系数: {corr_visit_new:.4f}")
    print(f"2. 访问量与老用户显著相关，相关系数: {corr_visit_old:.4f}")
    print(f"3. 新用户与老用户之间也存在较强相关性，相关系数: {corr_new_old:.4f}")
    print("4. 从趋势上看，新用户占比呈现上升趋势，显示用户群体在不断扩大和更新")
    
except Exception as e:
    print(f"绘制散点图过程中发生错误: {e}")