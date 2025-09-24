import pandas as pd
import numpy as np

# 示例数据：年龄与续保关系
data = {
    'age_group': ['18-30', '18-30', '18-30', '18-30', '18-30', 
                  '31-45', '31-45', '31-45', '31-45', '31-45',
                  '46-60', '46-60', '46-60', '46-60', '46-60',
                  '61+', '61+', '61+', '61+', '61+'],
    'renewal': ['No', 'No', 'Yes', 'Yes', 'Yes',  # 18-30: 2No, 3Yes
                'No', 'Yes', 'Yes', 'Yes', 'Yes',  # 31-45: 1No, 4Yes
                'Yes', 'Yes', 'Yes', 'Yes', 'Yes',  # 46-60: 0No, 5Yes
                'No', 'No', 'No', 'Yes', 'Yes']    # 61+: 3No, 2Yes
}

df = pd.DataFrame(data)

# 计算每个年龄段的续保情况
# 先将续保转换为数值
renewal_numeric = df['renewal'].map({'Yes': 1, 'No': 0})
age_stats = df.groupby('age_group').agg(
    total=('renewal', 'count'),
    renew_yes=(renewal_numeric, 'sum')
)
age_stats['renew_no'] = age_stats['total'] - age_stats['renew_yes']

# 计算总体统计量
total_yes = age_stats['renew_yes'].sum()
total_no = age_stats['renew_no'].sum()

print("=== 原始数据统计 ===")
print(age_stats)
print(f"总续保数(Yes): {total_yes}")
print(f"总不续保数(No): {total_no}")
print()

# 计算WOE和IV
age_stats['pct_yes'] = age_stats['renew_yes'] / total_yes
age_stats['pct_no'] = age_stats['renew_no'] / total_no

# 避免除零错误
age_stats['pct_yes'] = age_stats['pct_yes'].replace(0, 0.0001)
age_stats['pct_no'] = age_stats['pct_no'].replace(0, 0.0001)

# 计算WOE
age_stats['woe'] = np.log(age_stats['pct_no'] / age_stats['pct_yes'])

# 计算IV分量
age_stats['iv_component'] = (age_stats['pct_no'] - age_stats['pct_yes']) * age_stats['woe']

# 计算总IV
total_iv = age_stats['iv_component'].sum()

print("=== WOE和IV计算详情 ===")
print(age_stats[['total', 'renew_yes', 'renew_no', 'pct_yes', 'pct_no', 'woe', 'iv_component']])
print(f"\n总信息价值(IV): {total_iv:.4f}")
print()

# 解释每个年龄段的WOE含义
print("=== WOE值解释 ===")
for age_group, row in age_stats.iterrows():
    woe = row['woe']
    if woe > 0:
        meaning = "该年龄段不续保的可能性高于平均水平"
    elif woe < 0:
        meaning = "该年龄段续保的可能性高于平均水平"
    else:
        meaning = "该年龄段与平均水平无差异"
    
    print(f"{age_group}: WOE = {woe:.3f} - {meaning}")

print()
print("=== IV值解释 ===")
if total_iv < 0.02:
    iv_strength = "无预测能力"
elif total_iv < 0.1:
    iv_strength = "弱预测能力"
elif total_iv < 0.3:
    iv_strength = "中等预测能力"
elif total_iv < 0.5:
    iv_strength = "强预测能力"
else:
    iv_strength = "极强预测能力"

print(f"IV = {total_iv:.4f} - {iv_strength}")

# 显示详细计算过程
print("\n=== 详细计算过程（以18-30岁为例） ===")
age_18_30 = age_stats.loc['18-30']
print(f"18-30岁续保数: {age_18_30['renew_yes']}")
print(f"18-30岁不续保数: {age_18_30['renew_no']}")
print(f"续保比例: {age_18_30['pct_yes']:.4f} ({age_18_30['renew_yes']}/{total_yes})")
print(f"不续保比例: {age_18_30['pct_no']:.4f} ({age_18_30['renew_no']}/{total_no})")
print(f"WOE = ln({age_18_30['pct_no']:.4f}/{age_18_30['pct_yes']:.4f}) = ln({age_18_30['pct_no']/age_18_30['pct_yes']:.4f}) = {age_18_30['woe']:.4f}")
print(f"IV分量 = ({age_18_30['pct_no']:.4f} - {age_18_30['pct_yes']:.4f}) × {age_18_30['woe']:.4f} = {age_18_30['iv_component']:.4f}")