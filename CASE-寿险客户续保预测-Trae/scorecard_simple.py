import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 简化的WOE和IV计算函数
def calculate_woe_iv_simple(df, feature, target):
    """简化版的WOE和IV计算"""
    
    # 创建临时数据框
    temp_df = df[[feature, target]].copy()
    temp_df[target] = temp_df[target].map({'Yes': 1, 'No': 0})
    
    # 对连续变量进行分箱
    if temp_df[feature].dtype in ['int64', 'float64']:
        temp_df['bin'] = pd.qcut(temp_df[feature], q=5, duplicates='drop')
        grouped = temp_df.groupby('bin')[target].agg(['count', 'sum'])
    else:
        grouped = temp_df.groupby(feature)[target].agg(['count', 'sum'])
    
    grouped['non_event'] = grouped['count'] - grouped['sum']
    
    # 计算每个分箱的统计量
    total_event = grouped['sum'].sum()
    total_non_event = grouped['non_event'].sum()
    
    grouped['event_rate'] = grouped['sum'] / total_event
    grouped['non_event_rate'] = grouped['non_event'] / total_non_event
    
    # 避免除零错误
    grouped['non_event_rate'] = grouped['non_event_rate'].replace(0, 0.0001)
    grouped['event_rate'] = grouped['event_rate'].replace(0, 0.0001)
    
    # 计算WOE和IV
    grouped['woe'] = np.log(grouped['non_event_rate'] / grouped['event_rate'])
    grouped['iv'] = (grouped['non_event_rate'] - grouped['event_rate']) * grouped['woe']
    
    total_iv = grouped['iv'].sum()
    
    return grouped, total_iv

def create_scorecard():
    """创建评分卡模型"""
    
    print("=== 寿险客户续保评分卡模型 ===")
    
    # 1. 加载数据
    print("1. 加载数据...")
    df = pd.read_excel('policy_data.xlsx')
    
    # 2. 选择特征
    features = [
        'age', 'gender', 'income_level', 'education_level', 
        'occupation', 'marital_status', 'family_members',
        'policy_type', 'policy_term', 'premium_amount', 'claim_history'
    ]
    target = 'renewal'
    
    # 3. 计算IV值
    print("\n2. 计算特征IV值...")
    iv_results = {}
    
    for feature in features:
        try:
            _, iv_value = calculate_woe_iv_simple(df, feature, target)
            iv_results[feature] = iv_value
            print(f"{feature}: IV = {iv_value:.4f}")
        except Exception as e:
            print(f"计算 {feature} 的IV时出错: {e}")
            iv_results[feature] = 0
    
    # 4. 筛选IV >= 0.02的特征（降低阈值）
    print("\n3. 筛选重要特征(IV >= 0.02)...")
    selected_features = []
    for feature, iv in iv_results.items():
        if iv >= 0.02:
            selected_features.append(feature)
            print(f"✓ {feature}: IV = {iv:.4f} (保留)")
        else:
            print(f"✗ {feature}: IV = {iv:.4f} (剔除)")
    
    if not selected_features:
        print("没有特征被选中，使用所有特征")
        selected_features = features
    
    # 5. 准备特征矩阵
    print(f"\n4. 准备 {len(selected_features)} 个特征...")
    
    # 对分类变量进行编码
    X = df[selected_features].copy()
    y = df[target].map({'Yes': 1, 'No': 0})
    
    # 编码分类变量
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 6. 训练逻辑回归模型
    print("5. 训练逻辑回归模型...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练逻辑回归
    lr_model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # 7. 模型评估
    print("6. 模型评估...")
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 8. 构建评分卡
    print("\n7. 构建评分卡规则...")
    
    # 获取模型系数
    coefficients = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    
    # 计算每个特征的分数贡献
    feature_scores = {}
    for i, feature in enumerate(selected_features):
        feature_scores[feature] = coefficients[i]
    
    # 计算基础分数（通常设置为600分，odds为50:1时）
    base_score = 600
    pdo = 20  # 分数翻倍所需的odds变化
    
    # 计算分数转换参数
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(1/50)
    
    # 生成评分卡规则
    print("\n=== 评分卡计算规则 ===")
    print(f"基础分数: {base_score}")
    print(f"PDO (Points to Double Odds): {pdo}")
    print(f"转换公式: Score = {offset:.2f} + {factor:.2f} * log(odds)")
    
    print("\n=== 特征系数 ===")
    for feature, coef in feature_scores.items():
        print(f"{feature}: {coef:.4f}")
    
    # 9. 对测试数据进行预测
    print("\n8. 对测试数据进行预测...")
    test_df = pd.read_excel('policy_test.xlsx')
    
    # 准备测试特征
    X_test_new = test_df[selected_features].copy()
    
    # 编码分类变量 - 使用训练时的编码方式
    # 这里简化处理，实际应该保存训练时的编码器
    for col in X_test_new.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        # 拟合训练数据的编码
        train_values = df[col].unique()
        le.fit(train_values)
        # 处理测试集中可能的新类别，使用第一个类别
        X_test_new[col] = X_test_new[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X_test_new[col] = le.transform(X_test_new[col])
    
    # 标准化
    X_test_new_scaled = scaler.transform(X_test_new)
    
    # 预测概率
    probabilities = lr_model.predict_proba(X_test_new_scaled)[:, 1]
    
    # 转换为分数
    odds = probabilities / (1 - probabilities + 1e-10)
    scores = offset + factor * np.log(odds)
    
    # 生成预测结果
    results_df = pd.DataFrame({
        'policy_id': test_df['policy_id'],
        'renewal_probability': probabilities,
        'score': scores,
        'renewal': ['Yes' if p >= 0.5 else 'No' for p in probabilities]
    })
    
    # 保存结果
    results_df.to_csv('scorecard_predictions.csv', index=False, encoding='utf-8-sig')
    print("预测结果已保存到: scorecard_predictions.csv")
    
    # 显示分数分布
    print("\n=== 分数分布 ===")
    print(f"最低分: {scores.min():.2f}")
    print(f"最高分: {scores.max():.2f}")
    print(f"平均分: {scores.mean():.2f}")
    
    # 分数段分析
    score_bins = [0, 500, 600, 700, 800, 1000]
    score_labels = ['低风险', '中低风险', '中等风险', '中高风险', '高风险']
    
    results_df['risk_level'] = pd.cut(results_df['score'], bins=score_bins, labels=score_labels)
    print("\n风险等级分布:")
    print(results_df['risk_level'].value_counts())
    
    return lr_model, scaler, feature_scores

if __name__ == "__main__":
    create_scorecard()