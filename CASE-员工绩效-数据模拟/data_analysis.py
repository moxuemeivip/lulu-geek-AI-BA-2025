# 数据相关性分析脚本

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_data_correlations():
    """分析数据字段间的相关特征"""
    
    # 读取生成的Excel文件
    df = pd.read_excel('员工绩效数据分析.xlsx', sheet_name='员工绩效数据')
    
    print("🔍 数据相关性分析报告")
    print("=" * 50)
    
    # 1. 基础统计信息
    print("\n📊 基础统计信息:")
    print(f"总记录数: {len(df)}")
    print(f"员工数量: {df['employee_id'].nunique()}")
    print(f"时间跨度: {df['analysis_month'].min()} 至 {df['analysis_month'].max()}")
    
    # 2. 关键指标相关性分析
    print("\n🔗 关键指标相关性分析:")
    
    # 选择数值型字段进行相关性分析
    numeric_columns = [
        'new_customer_count', 'credit_card_issued', 'sales_completion_rate',
        'high_value_customer_count', 'customer_retention_rate', 
        'customer_satisfaction_score', 'bad_debt_rate', 'performance_score',
        'conversion_rate', 'work_days'
    ]
    
    correlation_matrix = df[numeric_columns].corr()
    
    print("\n📈 主要相关性关系:")
    
    # 分析绩效评分与其他指标的相关性
    performance_corr = correlation_matrix['performance_score'].sort_values(ascending=False)
    print("\n绩效评分相关性排序:")
    for col, corr in performance_corr.items():
        if col != 'performance_score':
            print(f"  {col}: {corr:.3f}")
    
    # 3. 业务逻辑验证
    print("\n✅ 业务逻辑验证:")
    
    # 验证发卡数量与新增客户数的关系
    card_customer_corr = df['credit_card_issued'].corr(df['new_customer_count'])
    print(f"发卡数量与新增客户数相关性: {card_customer_corr:.3f} (应该接近1)")
    
    # 验证销售完成率与绩效评分的关系
    sales_performance_corr = df['sales_completion_rate'].corr(df['performance_score'])
    print(f"销售完成率与绩效评分相关性: {sales_performance_corr:.3f} (应该为正相关)")
    
    # 验证不良率与绩效评分的关系
    bad_debt_performance_corr = df['bad_debt_rate'].corr(df['performance_score'])
    print(f"不良率与绩效评分相关性: {bad_debt_performance_corr:.3f} (应该为负相关)")
    
    # 4. 部门绩效对比
    print("\n🏢 部门绩效对比:")
    dept_performance = df.groupby('department').agg({
        'performance_score': ['mean', 'std'],
        'new_customer_count': 'mean',
        'sales_completion_rate': 'mean'
    }).round(2)
    print(dept_performance)
    
    # 5. 职位绩效对比
    print("\n👔 职位绩效对比:")
    position_performance = df.groupby('position').agg({
        'performance_score': ['mean', 'std'],
        'new_customer_count': 'mean',
        'sales_completion_rate': 'mean'
    }).round(2)
    print(position_performance)
    
    # 6. 月度趋势分析
    print("\n📅 月度趋势分析:")
    monthly_trend = df.groupby('analysis_month').agg({
        'performance_score': 'mean',
        'new_customer_count': 'sum',
        'sales_completion_rate': 'mean'
    }).round(2)
    print(monthly_trend)
    
    # 7. 异常值检测
    print("\n⚠️ 异常值检测:")
    
    # 检测绩效评分的异常值
    performance_mean = df['performance_score'].mean()
    performance_std = df['performance_score'].std()
    outliers = df[(df['performance_score'] < performance_mean - 2*performance_std) | 
                  (df['performance_score'] > performance_mean + 2*performance_std)]
    
    print(f"绩效评分异常值数量: {len(outliers)}")
    if len(outliers) > 0:
        print("异常值详情:")
        print(outliers[['employee_name', 'department', 'performance_score']].head())
    
    # 8. 数据质量检查
    print("\n🔍 数据质量检查:")
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("缺失值统计:")
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"  {col}: {missing}")
    
    # 检查数据范围
    print("\n数据范围检查:")
    print(f"绩效评分范围: {df['performance_score'].min():.2f} - {df['performance_score'].max():.2f}")
    print(f"销售完成率范围: {df['sales_completion_rate'].min():.2f}% - {df['sales_completion_rate'].max():.2f}%")
    print(f"新增客户数范围: {df['new_customer_count'].min()} - {df['new_customer_count'].max()}")
    
    return df, correlation_matrix

def create_correlation_heatmap(correlation_matrix):
    """创建相关性热力图"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('员工绩效指标相关性热力图', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('相关性热力图.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 相关性热力图已保存为: 相关性热力图.png")

if __name__ == "__main__":
    # 分析数据相关性
    df, corr_matrix = analyze_data_correlations()
    
    # 创建相关性热力图
    try:
        create_correlation_heatmap(corr_matrix)
    except Exception as e:
        print(f"⚠️ 无法生成热力图: {e}")
        print("请确保已安装 matplotlib 和 seaborn")
    
    print("\n✅ 数据分析完成！")
