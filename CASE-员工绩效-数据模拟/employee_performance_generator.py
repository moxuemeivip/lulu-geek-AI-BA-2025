# 银行信用卡中心员工绩效数据生成器

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EmployeePerformanceDataGenerator:
    """员工绩效数据生成器"""
    
    def __init__(self, num_employees=50, analysis_months=12):
        """
        初始化数据生成器
        
        Args:
            num_employees: 员工数量
            analysis_months: 分析月份数量
        """
        self.num_employees = num_employees
        self.analysis_months = analysis_months
        
        # 员工基础信息
        self.employee_names = [
            '张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十',
            '郑一', '王二', '冯三', '陈四', '褚五', '卫六', '蒋七', '沈八',
            '韩九', '杨十', '朱一', '秦二', '尤三', '许四', '何五', '吕六',
            '施七', '张八', '孔九', '曹十', '严一', '华二', '金三', '魏四',
            '陶五', '姜六', '戚七', '谢八', '邹九', '喻十', '柏一', '水二',
            '窦三', '章四', '云五', '苏六', '潘七', '葛八', '奚九', '范十',
            '彭一', '郎二'
        ]
        
        self.departments = ['信用卡营销部', '信用卡风控部', '信用卡运营部']
        self.positions = ['营销专员', '高级营销专员', '营销主管', '风控专员', '运营专员']
        
        # 设置随机种子确保结果可重现
        np.random.seed(42)
        random.seed(42)
    
    def generate_employee_basic_info(self):
        """生成员工基础信息"""
        employees = []
        
        for i in range(self.num_employees):
            employee_id = f"EMP{i+1:03d}"
            employee_name = self.employee_names[i]
            department = random.choice(self.departments)
            
            # 根据部门确定职位
            if department == '信用卡营销部':
                position = random.choice(['营销专员', '高级营销专员', '营销主管'])
            elif department == '信用卡风控部':
                position = '风控专员'
            else:
                position = '运营专员'
            
            # 生成入职日期（2020-2023年之间）
            hire_date = datetime(2020, 1, 1) + timedelta(
                days=random.randint(0, 1460)  # 4年内的随机日期
            )
            
            employees.append({
                'employee_id': employee_id,
                'employee_name': employee_name,
                'department': department,
                'position': position,
                'manager_id': f"MGR{random.randint(1, 5):03d}",
                'hire_date': hire_date.strftime('%Y-%m-%d')
            })
        
        return employees
    
    def generate_performance_data(self, employee_info):
        """生成绩效数据"""
        performance_data = []
        
        # 生成12个月的数据
        for month in range(1, self.analysis_months + 1):
            analysis_month = f"2024-{month:02d}"
            analysis_year = 2024
            analysis_quarter = f"2024-Q{(month-1)//3 + 1}"
            
            for emp in employee_info:
                # 根据职位和部门设置基础参数
                base_performance = self._get_base_performance(emp['position'], emp['department'])
                
                # 生成业务指标（考虑相关性）
                new_customer_count = self._generate_new_customers(base_performance)
                credit_card_issued = int(new_customer_count * random.uniform(0.8, 0.95))
                activation_rate = random.uniform(75, 90)  # 激活率75%-90%
                
                # 生成销售指标
                sales_target = random.uniform(800, 1200)  # 销售目标800-1200万
                sales_actual = sales_target * random.uniform(0.7, 1.2)  # 完成率70%-120%
                sales_completion_rate = (sales_actual / sales_target) * 100
                
                # 生成客户质量指标（与业务量正相关）
                high_value_customer_count = int(new_customer_count * random.uniform(0.2, 0.4))
                customer_retention_rate = random.uniform(85, 95)
                customer_satisfaction_score = random.uniform(4.0, 5.0)
                complaint_count = max(0, int(np.random.poisson(2)))  # 投诉次数服从泊松分布
                
                # 生成风险控制指标（与业务量负相关）
                bad_debt_rate = random.uniform(0.5, 2.5)
                risk_customer_count = max(0, int(np.random.poisson(3)))
                compliance_violation_count = max(0, int(np.random.poisson(0.5)))
                
                # 生成效率指标
                work_days = random.randint(20, 23)
                customer_contact_count = int(new_customer_count * random.uniform(3, 5))
                conversion_rate = random.uniform(20, 35)
                avg_deal_time = random.uniform(2, 5)
                
                # 计算综合绩效评分（考虑多个维度）
                performance_score = self._calculate_performance_score(
                    sales_completion_rate, customer_retention_rate, 
                    customer_satisfaction_score, bad_debt_rate, complaint_count
                )
                
                # 生成激励信息
                bonus_amount = max(0, (sales_actual - sales_target) * 100 + random.uniform(5000, 10000))
                commission_amount = sales_actual * random.uniform(0.3, 0.5)
                
                performance_data.append({
                    'employee_id': emp['employee_id'],
                    'employee_name': emp['employee_name'],
                    'department': emp['department'],
                    'position': emp['position'],
                    'manager_id': emp['manager_id'],
                    'hire_date': emp['hire_date'],
                    'analysis_month': analysis_month,
                    'analysis_year': analysis_year,
                    'analysis_quarter': analysis_quarter,
                    'new_customer_count': new_customer_count,
                    'new_customer_amount': round(new_customer_count * random.uniform(20, 35), 2),
                    'credit_card_issued': credit_card_issued,
                    'credit_card_amount': round(credit_card_issued * random.uniform(20, 30), 2),
                    'activation_rate': round(activation_rate, 2),
                    'sales_target': round(sales_target, 2),
                    'sales_actual': round(sales_actual, 2),
                    'sales_completion_rate': round(sales_completion_rate, 2),
                    'cross_sell_count': int(new_customer_count * random.uniform(0.4, 0.7)),
                    'cross_sell_amount': round(sales_actual * random.uniform(0.15, 0.25), 2),
                    'high_value_customer_count': high_value_customer_count,
                    'customer_retention_rate': round(customer_retention_rate, 2),
                    'customer_satisfaction_score': round(customer_satisfaction_score, 1),
                    'complaint_count': complaint_count,
                    'bad_debt_rate': round(bad_debt_rate, 2),
                    'risk_customer_count': risk_customer_count,
                    'compliance_violation_count': compliance_violation_count,
                    'work_days': work_days,
                    'customer_contact_count': customer_contact_count,
                    'conversion_rate': round(conversion_rate, 2),
                    'avg_deal_time': round(avg_deal_time, 2),
                    'performance_score': round(performance_score, 2),
                    'ranking_in_department': 0,  # 后续计算
                    'ranking_in_company': 0,     # 后续计算
                    'bonus_amount': round(bonus_amount, 2),
                    'commission_amount': round(commission_amount, 2),
                    'data_source': 'CRM系统',
                    'last_update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data_status': '有效'
                })
        
        return performance_data
    
    def _get_base_performance(self, position, department):
        """根据职位和部门获取基础绩效水平"""
        base_scores = {
            '营销主管': 1.2,
            '高级营销专员': 1.1,
            '营销专员': 1.0,
            '风控专员': 0.8,
            '运营专员': 0.9
        }
        return base_scores.get(position, 1.0)
    
    def _generate_new_customers(self, base_performance):
        """生成新增客户数（考虑基础绩效水平）"""
        # 基础客户数 + 随机波动
        base_customers = 30 * base_performance
        variation = random.uniform(0.7, 1.3)
        return max(10, int(base_customers * variation))
    
    def _calculate_performance_score(self, sales_completion_rate, retention_rate, 
                                   satisfaction_score, bad_debt_rate, complaint_count):
        """计算综合绩效评分"""
        # 销售完成率权重40%
        sales_score = min(100, sales_completion_rate) * 0.4
        
        # 客户留存率权重25%
        retention_score = retention_rate * 0.25
        
        # 客户满意度权重20%
        satisfaction_score = (satisfaction_score / 5) * 100 * 0.2
        
        # 风险控制权重15%（不良率越低越好）
        risk_score = max(0, (3 - bad_debt_rate) / 3 * 100) * 0.15
        
        # 投诉次数扣分
        complaint_penalty = min(10, complaint_count * 2)
        
        total_score = sales_score + retention_score + satisfaction_score + risk_score - complaint_penalty
        return max(0, min(100, total_score))
    
    def calculate_rankings(self, performance_data):
        """计算排名"""
        df = pd.DataFrame(performance_data)
        
        # 按月份分组计算排名
        for month in df['analysis_month'].unique():
            month_data = df[df['analysis_month'] == month]
            
            # 部门排名
            for dept in month_data['department'].unique():
                dept_data = month_data[month_data['department'] == dept]
                dept_rankings = dept_data['performance_score'].rank(method='dense', ascending=False)
                df.loc[(df['analysis_month'] == month) & (df['department'] == dept), 'ranking_in_department'] = dept_rankings
            
            # 公司排名
            company_rankings = month_data['performance_score'].rank(method='dense', ascending=False)
            df.loc[df['analysis_month'] == month, 'ranking_in_company'] = company_rankings
        
        return df.to_dict('records')
    
    def generate_excel_file(self, filename='员工绩效数据分析.xlsx'):
        """生成Excel文件"""
        print("正在生成员工基础信息...")
        employee_info = self.generate_employee_basic_info()
        
        print("正在生成绩效数据...")
        performance_data = self.generate_performance_data(employee_info)
        
        print("正在计算排名...")
        final_data = self.calculate_rankings(performance_data)
        
        print("正在导出Excel文件...")
        df = pd.DataFrame(final_data)
        
        # 创建Excel写入器
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 主数据表
            df.to_excel(writer, sheet_name='员工绩效数据', index=False)
            
            # 数据字典
            data_dict = self._create_data_dictionary()
            data_dict.to_excel(writer, sheet_name='数据字典', index=False)
            
            # 统计摘要
            summary = self._create_summary_statistics(df)
            summary.to_excel(writer, sheet_name='统计摘要', index=False)
        
        print(f"✅ 数据生成完成！文件已保存为: {filename}")
        print(f"📊 共生成 {len(final_data)} 条记录")
        print(f"👥 涉及 {self.num_employees} 名员工")
        print(f"📅 覆盖 {self.analysis_months} 个月的数据")
        
        return df
    
    def _create_data_dictionary(self):
        """创建数据字典"""
        data_dict = [
            {'字段名': 'employee_id', '字段类型': 'VARCHAR(20)', '字段说明': '员工工号', '示例值': 'EMP001'},
            {'字段名': 'employee_name', '字段类型': 'VARCHAR(50)', '字段说明': '员工姓名', '示例值': '张三'},
            {'字段名': 'department', '字段类型': 'VARCHAR(50)', '字段说明': '所属部门', '示例值': '信用卡营销部'},
            {'字段名': 'position', '字段类型': 'VARCHAR(50)', '字段说明': '职位', '示例值': '高级营销专员'},
            {'字段名': 'analysis_month', '字段类型': 'VARCHAR(7)', '字段说明': '分析月份', '示例值': '2024-01'},
            {'字段名': 'new_customer_count', '字段类型': 'INT', '字段说明': '新增客户数', '示例值': '45'},
            {'字段名': 'credit_card_issued', '字段类型': 'INT', '字段说明': '发卡数量', '示例值': '38'},
            {'字段名': 'sales_completion_rate', '字段类型': 'DECIMAL(5,2)', '字段说明': '销售完成率(%)', '示例值': '85.05'},
            {'字段名': 'performance_score', '字段类型': 'DECIMAL(5,2)', '字段说明': '综合绩效评分', '示例值': '88.50'},
            {'字段名': 'ranking_in_department', '字段类型': 'INT', '字段说明': '部门排名', '示例值': '5'}
        ]
        return pd.DataFrame(data_dict)
    
    def _create_summary_statistics(self, df):
        """创建统计摘要"""
        summary = []
        
        # 按部门统计
        dept_stats = df.groupby('department').agg({
            'performance_score': ['mean', 'std', 'min', 'max'],
            'new_customer_count': 'sum',
            'sales_actual': 'sum'
        }).round(2)
        
        summary.append({'指标': '部门统计', '详情': str(dept_stats)})
        
        # 按月份统计
        month_stats = df.groupby('analysis_month').agg({
            'performance_score': 'mean',
            'new_customer_count': 'sum'
        }).round(2)
        
        summary.append({'指标': '月度趋势', '详情': str(month_stats)})
        
        return pd.DataFrame(summary)

# 主程序
if __name__ == "__main__":
    # 创建数据生成器
    generator = EmployeePerformanceDataGenerator(
        num_employees=50,      # 50名员工
        analysis_months=12     # 12个月数据
    )
    
    # 生成Excel文件
    df = generator.generate_excel_file('员工绩效数据分析.xlsx')
    
    # 显示数据预览
    print("\n📋 数据预览:")
    print(df.head())
    
    print("\n📈 数据统计:")
    print(f"平均绩效评分: {df['performance_score'].mean():.2f}")
    print(f"平均新增客户数: {df['new_customer_count'].mean():.2f}")
    print(f"平均销售完成率: {df['sales_completion_rate'].mean():.2f}%")
