#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
员工绩效差异化分析 - 修复中文乱码版本
如果中文字体无法正常显示，则生成英文版本图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 尝试设置中文字体
def setup_chinese_font():
    """尝试设置中文字体"""
    try:
        # Windows系统字体设置
        font_candidates = [
            'Microsoft YaHei',
            'SimHei', 
            'SimSun',
            'KaiTi',
            'FangSong',
            'Microsoft JhengHei'
        ]
        
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font_name in font_candidates:
            if font_name in available_fonts:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 成功设置中文字体: {font_name}")
                return True
        
        print("⚠️ 未找到合适的中文字体，将使用英文标签")
        return False
        
    except Exception as e:
        print(f"❌ 设置中文字体失败: {e}")
        return False

# 设置字体
chinese_font_available = setup_chinese_font()

class PerformanceAnalysisFixed:
    """员工绩效差异化分析器 - 修复版本"""
    
    def __init__(self, file_path):
        """
        初始化分析器
        
        Args:
            file_path (str): Excel文件路径
        """
        self.file_path = file_path
        self.df = None
        self.use_chinese = chinese_font_available
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_excel(self.file_path, sheet_name='员工绩效数据')
            print(f"✅ 数据加载成功：{self.df.shape[0]} 行 × {self.df.shape[1]} 列")
        except Exception as e:
            print(f"❌ 数据加载失败：{e}")
            raise
    
    def get_labels(self, key):
        """获取标签（中文或英文）"""
        labels = {
            # 图表标题
            'performance_distribution_title': '员工绩效分布分析' if self.use_chinese else 'Employee Performance Distribution Analysis',
            'hierarchical_comparison_title': '绩效分层比较分析' if self.use_chinese else 'Hierarchical Performance Comparison Analysis',
            'correlation_analysis_title': '绩效与关键业务指标相关性分析' if self.use_chinese else 'Performance vs Key Business Metrics Correlation Analysis',
            
            # 轴标签
            'performance_level': '绩效等级' if self.use_chinese else 'Performance Level',
            'employee_count': '人数' if self.use_chinese else 'Employee Count',
            'average_performance': '平均绩效分数' if self.use_chinese else 'Average Performance Score',
            'performance_score': '绩效分数' if self.use_chinese else 'Performance Score',
            'frequency': '频次' if self.use_chinese else 'Frequency',
            'correlation_coefficient': '相关系数' if self.use_chinese else 'Correlation Coefficient',
            'customer_satisfaction': '客户满意度评分' if self.use_chinese else 'Customer Satisfaction Score',
            'first_call_resolution': '首次解决率 (%)' if self.use_chinese else 'First Call Resolution Rate (%)',
            
            # 图表标题
            'level_distribution': '绩效等级人数分布' if self.use_chinese else 'Performance Level Distribution',
            'level_proportion': '绩效等级比例分布' if self.use_chinese else 'Performance Level Proportion',
            'level_average': '各等级平均绩效分数' if self.use_chinese else 'Average Performance by Level',
            'score_histogram': '绩效分数分布直方图' if self.use_chinese else 'Performance Score Histogram',
            'department_comparison': '部门绩效对比' if self.use_chinese else 'Department Performance Comparison',
            'team_ranking': '团队绩效排名' if self.use_chinese else 'Team Performance Ranking',
            'position_comparison': '职级绩效对比' if self.use_chinese else 'Position Level Performance Comparison',
            'correlation_ranking': '绩效与业务指标相关性排名' if self.use_chinese else 'Performance vs Business Metrics Correlation Ranking',
            'satisfaction_vs_performance': '客户满意度 vs 绩效分数' if self.use_chinese else 'Customer Satisfaction vs Performance Score',
            'resolution_vs_performance': '首次解决率 vs 绩效分数' if self.use_chinese else 'First Call Resolution vs Performance Score',
            'correlation_heatmap': '关键指标相关性热力图' if self.use_chinese else 'Key Metrics Correlation Heatmap',
            
            # 图例
            'average_value': '平均值' if self.use_chinese else 'Average',
            'performance_levels': '绩效等级' if self.use_chinese else 'Performance Level',
            
            # 部门名称
            'dept1': '信用卡客服一部' if self.use_chinese else 'Credit Card Service Dept 1',
            'dept2': '信用卡客服二部' if self.use_chinese else 'Credit Card Service Dept 2', 
            'dept3': '信用卡客服三部' if self.use_chinese else 'Credit Card Service Dept 3',
            
            # 团队名称
            'team1': 'VIP客户服务组' if self.use_chinese else 'VIP Customer Service',
            'team2': '普通客户服务组' if self.use_chinese else 'Regular Customer Service',
            'team3': '投诉处理组' if self.use_chinese else 'Complaint Handling',
            'team4': '业务办理组' if self.use_chinese else 'Business Processing',
            
            # 职级名称
            'pos1': '高级客服专员' if self.use_chinese else 'Senior Customer Service Specialist',
            'pos2': '资深客服专员' if self.use_chinese else 'Experienced Customer Service Specialist',
            'pos3': '客服专员' if self.use_chinese else 'Customer Service Specialist',
            'pos4': '初级客服专员' if self.use_chinese else 'Junior Customer Service Specialist'
        }
        
        return labels.get(key, key)
    
    def create_performance_distribution_charts(self, level_counts, level_percentages, level_means):
        """创建绩效分布图表"""
        print(f"\n📈 生成绩效分布图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(self.get_labels('performance_distribution_title'), fontsize=16, fontweight='bold')
        
        # 1. 绩效等级人数分布柱状图
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        bars = axes[0, 0].bar(level_counts.index, level_counts.values, color=colors[:len(level_counts)])
        axes[0, 0].set_title(self.get_labels('level_distribution'), fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel(self.get_labels('performance_level'))
        axes[0, 0].set_ylabel(self.get_labels('employee_count'))
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, level_counts.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. 绩效等级比例饼图
        axes[0, 1].pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%', 
                      colors=colors[:len(level_counts)], startangle=90)
        axes[0, 1].set_title(self.get_labels('level_proportion'), fontsize=12, fontweight='bold')
        
        # 3. 各等级平均绩效分数对比
        bars = axes[1, 0].bar(level_means.index, level_means.values, color=colors[:len(level_means)])
        axes[1, 0].set_title(self.get_labels('level_average'), fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel(self.get_labels('performance_level'))
        axes[1, 0].set_ylabel(self.get_labels('average_performance'))
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, level_means.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 绩效分数分布直方图
        axes[1, 1].hist(self.df['performance_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(self.df['performance_score'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'{self.get_labels("average_value")}: {self.df["performance_score"].mean():.2f}')
        axes[1, 1].set_title(self.get_labels('score_histogram'), fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel(self.get_labels('performance_score'))
        axes[1, 1].set_ylabel(self.get_labels('frequency'))
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = 'Performance_Distribution_Analysis.png' if not self.use_chinese else '绩效分布分析_修复版.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 绩效分布图表已保存为：{filename}")
    
    def create_hierarchical_charts(self, dept_performance, team_performance, position_performance):
        """创建分层比较图表"""
        print(f"\n📈 生成分层比较图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(self.get_labels('hierarchical_comparison_title'), fontsize=16, fontweight='bold')
        
        # 1. 部门绩效对比
        dept_means = self.df.groupby('department')['performance_score'].mean()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        bars = axes[0, 0].bar(dept_means.index, dept_means.values, color=colors[:len(dept_means)])
        axes[0, 0].set_title(self.get_labels('department_comparison'), fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel(self.get_labels('average_performance'))
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
        axes[0, 1].set_title(self.get_labels('team_ranking'), fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel(self.get_labels('average_performance'))
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, team_performance['mean'])):
            axes[0, 1].text(value + 0.2, bar.get_y() + bar.get_height()/2, 
                           f'{value:.2f}', ha='left', va='center', fontweight='bold')
        
        # 3. 职级绩效对比
        bars = axes[1, 0].bar(position_performance.index, position_performance['mean'], 
                             color='lightcoral', alpha=0.7)
        axes[1, 0].set_title(self.get_labels('position_comparison'), fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel(self.get_labels('average_performance'))
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, position_performance['mean']):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 部门-团队热力图
        pivot_data = self.df.groupby(['department', 'team'])['performance_score'].mean().unstack(fill_value=0)
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Department-Team Performance Heatmap' if not self.use_chinese else '部门-团队绩效热力图', 
                           fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filename = 'Hierarchical_Performance_Comparison.png' if not self.use_chinese else '绩效分层比较分析_修复版.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 分层比较图表已保存为：{filename}")
    
    def create_correlation_charts(self, sorted_correlations):
        """创建相关性分析图表"""
        print(f"\n📈 生成相关性分析图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(self.get_labels('correlation_analysis_title'), fontsize=16, fontweight='bold')
        
        # 1. 相关性条形图
        metrics = [item[0] for item in sorted_correlations[:8]]  # 取前8个
        corr_values = [item[1] for item in sorted_correlations[:8]]
        colors = ['red' if x > 0 else 'blue' for x in corr_values]
        
        bars = axes[0, 0].barh(range(len(metrics)), corr_values, color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(len(metrics)))
        axes[0, 0].set_yticklabels(metrics)
        axes[0, 0].set_xlabel(self.get_labels('correlation_coefficient'))
        axes[0, 0].set_title(self.get_labels('correlation_ranking'), fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, corr_values)):
            axes[0, 0].text(value + 0.01 if value > 0 else value - 0.01, 
                           bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                           ha='left' if value > 0 else 'right', va='center', fontweight='bold')
        
        # 2. 客户满意度 vs 绩效散点图
        axes[0, 1].scatter(self.df['customer_satisfaction_score'], self.df['performance_score'], 
                           alpha=0.6, color='blue')
        axes[0, 1].set_xlabel(self.get_labels('customer_satisfaction'))
        axes[0, 1].set_ylabel(self.get_labels('performance_score'))
        axes[0, 1].set_title(self.get_labels('satisfaction_vs_performance'), fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(self.df['customer_satisfaction_score'], self.df['performance_score'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.df['customer_satisfaction_score'], p(self.df['customer_satisfaction_score']), 
                       "r--", alpha=0.8, linewidth=2)
        
        # 3. 首次解决率 vs 绩效散点图
        axes[1, 0].scatter(self.df['first_call_resolution_rate'], self.df['performance_score'], 
                         alpha=0.6, color='green')
        axes[1, 0].set_xlabel(self.get_labels('first_call_resolution'))
        axes[1, 0].set_ylabel(self.get_labels('performance_score'))
        axes[1, 0].set_title(self.get_labels('resolution_vs_performance'), fontsize=12, fontweight='bold')
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
        axes[1, 1].set_title(self.get_labels('correlation_heatmap'), fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filename = 'Performance_Correlation_Analysis.png' if not self.use_chinese else '绩效相关性分析_修复版.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 相关性分析图表已保存为：{filename}")
    
    def run_analysis(self):
        """运行完整分析"""
        print(f"\n🚀 开始员工绩效差异化分析（修复版）")
        print(f"=" * 80)
        
        # 分析绩效分布
        level_counts = self.df['performance_level'].value_counts()
        level_percentages = self.df['performance_level'].value_counts(normalize=True) * 100
        level_means = self.df.groupby('performance_level')['performance_score'].mean()
        
        print(f"\n📊 绩效等级分布：")
        for level in sorted(level_counts.index):
            count = level_counts[level]
            percentage = level_percentages[level]
            mean_score = level_means[level]
            print(f"   {level}级：{count} 人 ({percentage:.1f}%)，平均绩效 {mean_score:.2f} 分")
        
        # 创建绩效分布图表
        self.create_performance_distribution_charts(level_counts, level_percentages, level_means)
        
        # 分析分层比较
        dept_performance = self.df.groupby('department')['performance_score'].agg(['mean', 'std', 'count']).round(2)
        team_performance = self.df.groupby('team')['performance_score'].agg(['mean', 'std', 'count']).round(2).sort_values('mean', ascending=False)
        position_performance = self.df.groupby('position')['performance_score'].agg(['mean', 'std', 'count']).round(2).sort_values('mean', ascending=False)
        
        print(f"\n📊 部门绩效排名：")
        for i, (dept, row) in enumerate(dept_performance.iterrows(), 1):
            print(f"   {i}. {dept}：{row['mean']:.2f} 分")
        
        print(f"\n📊 团队绩效排名：")
        for i, (team, row) in enumerate(team_performance.iterrows(), 1):
            print(f"   {i}. {team}：{row['mean']:.2f} 分")
        
        print(f"\n📊 职级绩效排名：")
        for i, (position, row) in enumerate(position_performance.iterrows(), 1):
            print(f"   {i}. {position}：{row['mean']:.2f} 分")
        
        # 创建分层比较图表
        self.create_hierarchical_charts(dept_performance, team_performance, position_performance)
        
        # 分析相关性
        key_metrics = [
            'efficiency_score', 'quality_score', 'professional_score', 'attitude_score',
            'customer_satisfaction_score', 'first_call_resolution_rate', 'attendance_rate',
            'call_count', 'knowledge_test_score', 'training_hours', 'complaint_count',
            'business_completion_count', 'business_error_rate', 'online_time_rate'
        ]
        
        correlations = {}
        for metric in key_metrics:
            if metric in self.df.columns:
                corr = self.df[metric].corr(self.df['performance_score'])
                correlations[metric] = corr
        
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n📈 绩效与业务指标相关性分析：")
        for metric, corr in sorted_correlations[:8]:
            strength = "强" if abs(corr) > 0.5 else "中等" if abs(corr) > 0.3 else "弱"
            direction = "正相关" if corr > 0 else "负相关"
            print(f"   {metric}：{corr:.3f} ({strength}{direction})")
        
        # 创建相关性分析图表
        self.create_correlation_charts(sorted_correlations)
        
        print(f"\n🎉 分析完成！")
        print(f"=" * 80)
        print(f"📁 生成的文件：")
        if self.use_chinese:
            print(f"   - 绩效分布分析_修复版.png")
            print(f"   - 绩效分层比较分析_修复版.png")
            print(f"   - 绩效相关性分析_修复版.png")
        else:
            print(f"   - Performance_Distribution_Analysis.png")
            print(f"   - Hierarchical_Performance_Comparison.png")
            print(f"   - Performance_Correlation_Analysis.png")
        
        return {
            'level_counts': level_counts,
            'level_percentages': level_percentages,
            'level_means': level_means,
            'dept_performance': dept_performance,
            'team_performance': team_performance,
            'position_performance': position_performance,
            'sorted_correlations': sorted_correlations
        }

def main():
    """主函数"""
    try:
        # 创建分析器
        analyzer = PerformanceAnalysisFixed('员工绩效.xlsx')
        
        # 运行完整分析
        results = analyzer.run_analysis()
        
        print(f"\n✅ 所有分析完成！")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误：{e}")
        raise

if __name__ == "__main__":
    main()
