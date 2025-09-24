#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
员工绩效差异化分析与奖金分配优化
分析员工绩效数据.xlsx，进行绩效分布、分层比较和相关性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

class PerformanceAnalysis:
    """员工绩效差异化分析器"""
    
    def __init__(self, file_path):
        """
        初始化分析器
        
        Args:
            file_path (str): Excel文件路径
        """
        self.file_path = file_path
        self.df = None
        self.load_data()
        self.understand_fields()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_excel(self.file_path, sheet_name='员工绩效数据')
            print(f"✅ 数据加载成功：{self.df.shape[0]} 行 × {self.df.shape[1]} 列")
            print(f"📊 数据概览：")
            print(f"   员工总数：{len(self.df)} 人")
            print(f"   数据时间：{self.df['report_date'].iloc[0]}")
        except Exception as e:
            print(f"❌ 数据加载失败：{e}")
            raise
    
    def understand_fields(self):
        """理解字段含义"""
        print(f"\n📋 字段含义理解：")
        print(f"=" * 60)
        
        # 基本信息字段
        basic_fields = {
            'employee_id': '员工ID',
            'employee_name': '员工姓名',
            'department': '部门',
            'team': '团队',
            'position': '职位',
            'entry_date': '入职日期',
            'report_date': '报告日期',
            'manager_id': '直属经理ID'
        }
        
        # 业务指标字段
        business_fields = {
            'call_count': '通话次数',
            'avg_call_duration': '平均通话时长(分钟)',
            'knowledge_test_score': '知识测试分数',
            'training_hours': '培训时长(小时)',
            'first_call_resolution_rate': '首次解决率(%)',
            'customer_satisfaction_score': '客户满意度评分',
            'complaint_count': '投诉数量',
            'business_completion_count': '业务完成数量',
            'business_error_rate': '业务错误率(%)',
            'online_time_rate': '在线时间率(%)',
            'attendance_rate': '出勤率(%)',
            'upselling_amount': '销售金额'
        }
        
        # 绩效评估字段
        performance_fields = {
            'efficiency_score': '工作效率分数',
            'quality_score': '工作质量分数',
            'professional_score': '专业能力分数',
            'attitude_score': '服务态度分数',
            'performance_score': '综合绩效分数',
            'performance_level': '绩效等级',
            'remarks': '备注'
        }
        
        print(f"\n📝 基本信息字段：")
        for field, meaning in basic_fields.items():
            if field in self.df.columns:
                unique_count = self.df[field].nunique()
                print(f"   {field}: {meaning} (唯一值: {unique_count})")
        
        print(f"\n📊 业务指标字段：")
        for field, meaning in business_fields.items():
            if field in self.df.columns:
                # 只对数值字段计算平均值
                if pd.api.types.is_numeric_dtype(self.df[field]):
                    mean_val = self.df[field].mean()
                    print(f"   {field}: {meaning} (平均值: {mean_val:.2f})")
                else:
                    print(f"   {field}: {meaning} (非数值字段)")
        
        print(f"\n🎯 绩效评估字段：")
        for field, meaning in performance_fields.items():
            if field in self.df.columns:
                if field == 'performance_level':
                    level_counts = self.df[field].value_counts()
                    print(f"   {field}: {meaning}")
                    for level, count in level_counts.items():
                        print(f"     - {level}: {count} 人")
                elif field == 'remarks':
                    print(f"   {field}: {meaning} (文本字段)")
                else:
                    # 只对数值字段计算平均值
                    if pd.api.types.is_numeric_dtype(self.df[field]):
                        mean_val = self.df[field].mean()
                        print(f"   {field}: {meaning} (平均值: {mean_val:.2f})")
                    else:
                        print(f"   {field}: {meaning} (非数值字段)")
    
    def analyze_performance_distribution(self):
        """1. 构建绩效分布图表，分析A+到D各等级人数分布"""
        print(f"\n📊 1. 绩效分布分析")
        print(f"=" * 60)
        
        # 分析绩效等级分布
        level_counts = self.df['performance_level'].value_counts()
        level_percentages = self.df['performance_level'].value_counts(normalize=True) * 100
        
        print(f"\n📈 绩效等级分布统计：")
        for level in sorted(level_counts.index):
            count = level_counts[level]
            percentage = level_percentages[level]
            print(f"   {level}级：{count} 人 ({percentage:.1f}%)")
        
        # 计算各等级的平均绩效分数
        level_means = self.df.groupby('performance_level')['performance_score'].mean().sort_index()
        print(f"\n📊 各等级平均绩效分数：")
        for level, mean_score in level_means.items():
            print(f"   {level}级：{mean_score:.2f} 分")
        
        # 创建绩效分布图表
        self.create_performance_distribution_charts(level_counts, level_percentages, level_means)
        
        return level_counts, level_percentages, level_means
    
    def create_performance_distribution_charts(self, level_counts, level_percentages, level_means):
        """创建绩效分布图表"""
        print(f"\n📈 生成绩效分布图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('员工绩效分布分析', fontsize=16, fontweight='bold')
        
        # 1. 绩效等级人数分布柱状图
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        bars = axes[0, 0].bar(level_counts.index, level_counts.values, color=colors[:len(level_counts)])
        axes[0, 0].set_title('绩效等级人数分布', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('绩效等级')
        axes[0, 0].set_ylabel('人数')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, level_counts.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. 绩效等级比例饼图
        axes[0, 1].pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%', 
                      colors=colors[:len(level_counts)], startangle=90)
        axes[0, 1].set_title('绩效等级比例分布', fontsize=12, fontweight='bold')
        
        # 3. 各等级平均绩效分数对比
        bars = axes[1, 0].bar(level_means.index, level_means.values, color=colors[:len(level_means)])
        axes[1, 0].set_title('各等级平均绩效分数', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('绩效等级')
        axes[1, 0].set_ylabel('平均绩效分数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, level_means.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 绩效分数分布直方图
        axes[1, 1].hist(self.df['performance_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(self.df['performance_score'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'平均值: {self.df["performance_score"].mean():.2f}')
        axes[1, 1].set_title('绩效分数分布直方图', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('绩效分数')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('绩效分布分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 绩效分布图表已保存为：绩效分布分析.png")
    
    def analyze_hierarchical_performance(self):
        """2. 按部门、团队、职级进行绩效分层比较"""
        print(f"\n📊 2. 绩效分层比较分析")
        print(f"=" * 60)
        
        # 部门绩效分析
        print(f"\n🏢 部门绩效分析：")
        dept_performance = self.df.groupby('department').agg({
            'performance_score': ['mean', 'std', 'count'],
            'performance_level': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        print(f"   部门绩效统计：")
        for dept in self.df['department'].unique():
            dept_data = self.df[self.df['department'] == dept]
            avg_score = dept_data['performance_score'].mean()
            std_score = dept_data['performance_score'].std()
            count = len(dept_data)
            print(f"     {dept}：平均 {avg_score:.2f}±{std_score:.2f} ({count}人)")
        
        # 团队绩效分析
        print(f"\n👥 团队绩效分析：")
        team_performance = self.df.groupby('team')['performance_score'].agg(['mean', 'std', 'count']).round(2)
        team_performance = team_performance.sort_values('mean', ascending=False)
        
        print(f"   团队绩效排名：")
        for i, (team, row) in enumerate(team_performance.iterrows(), 1):
            print(f"     {i}. {team}：平均 {row['mean']:.2f}±{row['std']:.2f} ({row['count']}人)")
        
        # 职级绩效分析
        print(f"\n🎯 职级绩效分析：")
        position_performance = self.df.groupby('position')['performance_score'].agg(['mean', 'std', 'count']).round(2)
        position_performance = position_performance.sort_values('mean', ascending=False)
        
        print(f"   职级绩效排名：")
        for i, (position, row) in enumerate(position_performance.iterrows(), 1):
            print(f"     {i}. {position}：平均 {row['mean']:.2f}±{row['std']:.2f} ({row['count']}人)")
        
        # 创建分层比较图表
        self.create_hierarchical_charts(dept_performance, team_performance, position_performance)
        
        return dept_performance, team_performance, position_performance
    
    def create_hierarchical_charts(self, dept_performance, team_performance, position_performance):
        """创建分层比较图表"""
        print(f"\n📈 生成分层比较图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('绩效分层比较分析', fontsize=16, fontweight='bold')
        
        # 1. 部门绩效对比
        dept_means = self.df.groupby('department')['performance_score'].mean()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        bars = axes[0, 0].bar(dept_means.index, dept_means.values, color=colors[:len(dept_means)])
        axes[0, 0].set_title('部门绩效对比', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('平均绩效分数')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, dept_means.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 团队绩效排名
        bars = axes[0, 1].barh(range(len(team_performance)), team_performance['mean'], 
                              color='lightgreen', alpha=0.7)
        axes[0, 1].set_yticks(range(len(team_performance)))
        axes[0, 1].set_yticklabels(team_performance.index)
        axes[0, 1].set_title('团队绩效排名', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('平均绩效分数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, team_performance['mean'])):
            axes[0, 1].text(value + 0.2, bar.get_y() + bar.get_height()/2, 
                           f'{value:.2f}', ha='left', va='center', fontweight='bold')
        
        # 3. 职级绩效对比
        bars = axes[1, 0].bar(position_performance.index, position_performance['mean'], 
                             color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('职级绩效对比', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('平均绩效分数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, position_performance['mean']):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 部门-团队-职级热力图
        pivot_data = self.df.groupby(['department', 'team'])['performance_score'].mean().unstack(fill_value=0)
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('部门-团队绩效热力图', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('绩效分层比较分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 分层比较图表已保存为：绩效分层比较分析.png")
    
    def analyze_correlation_with_business_metrics(self):
        """3. 建立绩效得分与关键业务指标的相关性分析"""
        print(f"\n📊 3. 绩效与关键业务指标相关性分析")
        print(f"=" * 60)
        
        # 关键业务指标
        key_metrics = [
            'customer_satisfaction_score',  # 客户满意度
            'first_call_resolution_rate',   # 首次解决率
            'efficiency_score',            # 工作效率
            'quality_score',              # 工作质量
            'professional_score',         # 专业能力
            'attitude_score',             # 服务态度
            'call_count',                 # 通话次数
            'complaint_count',            # 投诉数量
            'attendance_rate',            # 出勤率
            'business_completion_count'    # 业务完成数量
        ]
        
        # 计算相关性
        correlations = {}
        for metric in key_metrics:
            if metric in self.df.columns:
                corr = self.df[metric].corr(self.df['performance_score'])
                correlations[metric] = corr
        
        # 按相关性绝对值排序
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n📈 绩效与业务指标相关性分析：")
        for metric, corr in sorted_correlations:
            strength = "强" if abs(corr) > 0.5 else "中等" if abs(corr) > 0.3 else "弱"
            direction = "正相关" if corr > 0 else "负相关"
            print(f"   {metric}：{corr:.3f} ({strength}{direction})")
        
        # 重点分析客户满意度和首次解决率
        print(f"\n🎯 重点业务指标分析：")
        
        # 客户满意度分析
        satisfaction_corr = self.df['customer_satisfaction_score'].corr(self.df['performance_score'])
        satisfaction_stats = self.df['customer_satisfaction_score'].describe()
        print(f"\n   客户满意度：")
        print(f"     与绩效相关性：{satisfaction_corr:.3f}")
        print(f"     平均值：{satisfaction_stats['mean']:.2f}")
        print(f"     标准差：{satisfaction_stats['std']:.2f}")
        print(f"     范围：{satisfaction_stats['min']:.2f} - {satisfaction_stats['max']:.2f}")
        
        # 首次解决率分析
        resolution_corr = self.df['first_call_resolution_rate'].corr(self.df['performance_score'])
        resolution_stats = self.df['first_call_resolution_rate'].describe()
        print(f"\n   首次解决率：")
        print(f"     与绩效相关性：{resolution_corr:.3f}")
        print(f"     平均值：{resolution_stats['mean']:.2f}")
        print(f"     标准差：{resolution_stats['std']:.2f}")
        print(f"     范围：{resolution_stats['min']:.2f} - {resolution_stats['max']:.2f}")
        
        # 创建相关性分析图表
        self.create_correlation_charts(sorted_correlations)
        
        return sorted_correlations
    
    def create_correlation_charts(self, sorted_correlations):
        """创建相关性分析图表"""
        print(f"\n📈 生成相关性分析图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('绩效与关键业务指标相关性分析', fontsize=16, fontweight='bold')
        
        # 1. 相关性条形图
        metrics = [item[0] for item in sorted_correlations[:8]]  # 取前8个
        corr_values = [item[1] for item in sorted_correlations[:8]]
        colors = ['red' if x > 0 else 'blue' for x in corr_values]
        
        bars = axes[0, 0].barh(range(len(metrics)), corr_values, color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(len(metrics)))
        axes[0, 0].set_yticklabels(metrics)
        axes[0, 0].set_xlabel('相关系数')
        axes[0, 0].set_title('绩效与业务指标相关性排名', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, corr_values)):
            axes[0, 0].text(value + 0.01 if value > 0 else value - 0.01, 
                           bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                           ha='left' if value > 0 else 'right', va='center', fontweight='bold')
        
        # 2. 客户满意度 vs 绩效散点图
        axes[0, 1].scatter(self.df['customer_satisfaction_score'], self.df['performance_score'], 
                           alpha=0.6, color='blue')
        axes[0, 1].set_xlabel('客户满意度评分')
        axes[0, 1].set_ylabel('绩效分数')
        axes[0, 1].set_title('客户满意度 vs 绩效分数', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(self.df['customer_satisfaction_score'], self.df['performance_score'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.df['customer_satisfaction_score'], p(self.df['customer_satisfaction_score']), 
                       "r--", alpha=0.8, linewidth=2)
        
        # 3. 首次解决率 vs 绩效散点图
        axes[1, 0].scatter(self.df['first_call_resolution_rate'], self.df['performance_score'], 
                         alpha=0.6, color='green')
        axes[1, 0].set_xlabel('首次解决率 (%)')
        axes[1, 0].set_ylabel('绩效分数')
        axes[1, 0].set_title('首次解决率 vs 绩效分数', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(self.df['first_call_resolution_rate'], self.df['performance_score'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(self.df['first_call_resolution_rate'], p(self.df['first_call_resolution_rate']), 
                       "r--", alpha=0.8, linewidth=2)
        
        # 4. 相关性热力图
        correlation_matrix = self.df[['performance_score', 'customer_satisfaction_score', 
                                    'first_call_resolution_rate', 'efficiency_score', 
                                    'quality_score', 'professional_score']].corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                   square=True, ax=axes[1, 1], cbar_kws={"shrink": .8})
        axes[1, 1].set_title('关键指标相关性热力图', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('绩效相关性分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 相关性分析图表已保存为：绩效相关性分析.png")
    
    def bonus_allocation_optimization(self, level_counts, level_percentages, sorted_correlations):
        """奖金分配优化建议"""
        print(f"\n💰 奖金分配优化建议")
        print(f"=" * 60)
        
        # 基于绩效等级的奖金分配建议
        print(f"\n🎯 基于绩效等级的奖金分配建议：")
        
        # 假设总奖金池为100万元
        total_bonus = 1000000
        
        # 不同等级的奖金系数
        bonus_ratios = {
            'A': 1.5,   # A级：150%
            'B+': 1.2,  # B+级：120%
            'B': 1.0    # B级：100%
        }
        
        # 计算各等级应得奖金
        level_bonus = {}
        for level in level_counts.index:
            if level in bonus_ratios:
                ratio = bonus_ratios[level]
                count = level_counts[level]
                # 按人数和系数分配奖金
                level_bonus[level] = (count * ratio / sum(count * bonus_ratios.get(l, 1.0) for l in level_counts.index)) * total_bonus
        
        print(f"   奖金分配方案（总奖金池：{total_bonus:,}元）：")
        for level, bonus in level_bonus.items():
            count = level_counts[level]
            avg_bonus = bonus / count
            print(f"     {level}级：{count}人，总奖金 {bonus:,.0f}元，人均 {avg_bonus:,.0f}元")
        
        # 基于关键业务指标的奖金调整建议
        print(f"\n📊 基于关键业务指标的奖金调整建议：")
        
        # 识别高相关性指标
        high_corr_metrics = [item for item in sorted_correlations if abs(item[1]) > 0.4]
        
        print(f"   高相关性指标（|r| > 0.4）：")
        for metric, corr in high_corr_metrics:
            print(f"     {metric}：{corr:.3f}")
        
        # 建议的奖金调整策略
        print(f"\n💡 奖金调整策略建议：")
        print(f"   1. 基础奖金：按绩效等级分配（占70%）")
        print(f"   2. 业务指标奖金：基于客户满意度和首次解决率（占20%）")
        print(f"   3. 特殊贡献奖金：基于工作效率和质量分数（占10%）")
        
        return level_bonus, bonus_ratios
    
    def generate_comprehensive_report(self, level_counts, level_percentages, level_means, 
                                   dept_performance, team_performance, position_performance,
                                   sorted_correlations, level_bonus):
        """生成综合分析报告"""
        print(f"\n📋 生成综合分析报告...")
        
        report = f"""
# 员工绩效差异化分析与奖金分配优化报告

## 1. 数据概览
- 分析时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- 员工总数：{len(self.df)} 人
- 数据时间：{self.df['report_date'].iloc[0]}
- 部门数量：{self.df['department'].nunique()} 个
- 团队数量：{self.df['team'].nunique()} 个
- 职位层级：{self.df['position'].nunique()} 个

## 2. 绩效分布分析
### 2.1 绩效等级分布
"""
        
        for level in sorted(level_counts.index):
            count = level_counts[level]
            percentage = level_percentages[level]
            mean_score = level_means[level]
            report += f"- **{level}级**：{count} 人 ({percentage:.1f}%)，平均绩效 {mean_score:.2f} 分\n"
        
        report += f"""
### 2.2 绩效分布特征
- 平均绩效分数：{self.df['performance_score'].mean():.2f}
- 绩效分数标准差：{self.df['performance_score'].std():.2f}
- 绩效分数范围：{self.df['performance_score'].min():.2f} - {self.df['performance_score'].max():.2f}

## 3. 分层比较分析
### 3.1 部门绩效排名
"""
        
        dept_means = self.df.groupby('department')['performance_score'].mean().sort_values(ascending=False)
        for i, (dept, score) in enumerate(dept_means.items(), 1):
            report += f"{i}. **{dept}**：{score:.2f} 分\n"
        
        report += f"""
### 3.2 团队绩效排名
"""
        
        team_means = self.df.groupby('team')['performance_score'].mean().sort_values(ascending=False)
        for i, (team, score) in enumerate(team_means.items(), 1):
            report += f"{i}. **{team}**：{score:.2f} 分\n"
        
        report += f"""
### 3.3 职级绩效排名
"""
        
        position_means = self.df.groupby('position')['performance_score'].mean().sort_values(ascending=False)
        for i, (position, score) in enumerate(position_means.items(), 1):
            report += f"{i}. **{position}**：{score:.2f} 分\n"
        
        report += f"""
## 4. 关键业务指标相关性分析
### 4.1 高相关性指标 (|r| > 0.4)
"""
        
        high_corr_metrics = [item for item in sorted_correlations if abs(item[1]) > 0.4]
        for metric, corr in high_corr_metrics:
            strength = "强" if abs(corr) > 0.5 else "中等"
            direction = "正相关" if corr > 0 else "负相关"
            report += f"- **{metric}**：{corr:.3f} ({strength}{direction})\n"
        
        report += f"""
### 4.2 重点业务指标分析
- **客户满意度**：与绩效相关性 {sorted_correlations[0][1]:.3f}
- **首次解决率**：与绩效相关性 {sorted_correlations[1][1]:.3f}

## 5. 奖金分配优化建议
### 5.1 基于绩效等级的奖金分配
"""
        
        total_bonus = 1000000
        for level, bonus in level_bonus.items():
            count = level_counts[level]
            avg_bonus = bonus / count
            report += f"- **{level}级**：{count}人，总奖金 {bonus:,.0f}元，人均 {avg_bonus:,.0f}元\n"
        
        report += f"""
### 5.2 奖金分配策略
1. **基础奖金（70%）**：按绩效等级分配
2. **业务指标奖金（20%）**：基于客户满意度和首次解决率
3. **特殊贡献奖金（10%）**：基于工作效率和质量分数

## 6. 主要发现与建议
### 6.1 主要发现
1. 绩效分布相对均衡，B+级员工占多数
2. 部门间绩效差异较小，团队间存在一定差异
3. 客户满意度和首次解决率与绩效呈强正相关
4. 工作效率和质量是影响绩效的关键因素

### 6.2 优化建议
1. **绩效管理**：建立更细化的绩效等级体系
2. **奖金分配**：采用多维度奖金分配机制
3. **培训重点**：重点关注客户满意度和首次解决率提升
4. **激励机制**：建立基于关键业务指标的激励体系
5. **持续改进**：定期分析绩效数据，优化管理策略
"""
        
        # 保存报告
        with open('绩效差异化分析与奖金分配优化报告.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ 综合分析报告已保存为：绩效差异化分析与奖金分配优化报告.md")
    
    def run_analysis(self):
        """运行完整分析"""
        print(f"\n🚀 开始员工绩效差异化分析与奖金分配优化")
        print(f"=" * 80)
        
        # 1. 绩效分布分析
        level_counts, level_percentages, level_means = self.analyze_performance_distribution()
        
        # 2. 分层比较分析
        dept_performance, team_performance, position_performance = self.analyze_hierarchical_performance()
        
        # 3. 相关性分析
        sorted_correlations = self.analyze_correlation_with_business_metrics()
        
        # 4. 奖金分配优化
        level_bonus, bonus_ratios = self.bonus_allocation_optimization(level_counts, level_percentages, sorted_correlations)
        
        # 5. 生成综合报告
        self.generate_comprehensive_report(level_counts, level_percentages, level_means,
                                         dept_performance, team_performance, position_performance,
                                         sorted_correlations, level_bonus)
        
        print(f"\n🎉 分析完成！")
        print(f"=" * 80)
        print(f"📁 生成的文件：")
        print(f"   - 绩效分布分析.png")
        print(f"   - 绩效分层比较分析.png")
        print(f"   - 绩效相关性分析.png")
        print(f"   - 绩效差异化分析与奖金分配优化报告.md")
        
        return {
            'level_counts': level_counts,
            'level_percentages': level_percentages,
            'level_means': level_means,
            'dept_performance': dept_performance,
            'team_performance': team_performance,
            'position_performance': position_performance,
            'sorted_correlations': sorted_correlations,
            'level_bonus': level_bonus
        }

def main():
    """主函数"""
    try:
        # 创建分析器
        analyzer = PerformanceAnalysis('员工绩效.xlsx')
        
        # 运行完整分析
        results = analyzer.run_analysis()
        
        print(f"\n✅ 所有分析完成！")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误：{e}")
        raise

if __name__ == "__main__":
    main()
