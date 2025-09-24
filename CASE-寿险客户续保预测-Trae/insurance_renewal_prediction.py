import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 1. 数据加载和预处理
def load_and_preprocess_data():
    # 加载训练数据
    train_df = pd.read_excel('policy_data.xlsx')
    
    # 加载测试数据
    test_df = pd.read_excel('policy_test.xlsx')
    
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    
    # 保存测试数据的ID
    test_ids = test_df['policy_id']
    
    # 合并数据以便统一处理
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 特征工程 - 创建新特征
    combined_df = create_new_features(combined_df)
    
    # 分离回训练和测试数据
    train_processed = combined_df.iloc[:len(train_df)]
    test_processed = combined_df.iloc[len(train_df):]
    
    return train_processed, test_processed, test_ids

def create_new_features(df):
    """创建新的衍生特征"""
    
    # 1. 保单持续时间（天）
    df['policy_duration_days'] = (df['policy_end_date'] - df['policy_start_date']).dt.days
    
    # 2. 年龄分段
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 35, 50, 70], 
                            labels=['青年', '中年', '中老年'])
    
    # 3. 保费收入比（需要将收入水平转换为数值）
    income_mapping = {'低': 1, '中': 2, '高': 3}
    df['income_numeric'] = df['income_level'].map(income_mapping)
    df['premium_income_ratio'] = df['premium_amount'] / df['income_numeric']
    
    # 4. 职业稳定性编码
    occupation_stability = {
        '医生': 3, '律师': 3, '工程师': 3,  # 高稳定性职业
        '经理': 2, '设计师': 2,           # 中等稳定性职业
        '销售': 1                         # 低稳定性职业
    }
    df['occupation_stability'] = df['occupation'].map(occupation_stability)
    
    # 5. 家庭责任指数（基于年龄、婚姻状况、家庭成员数量）
    marital_weight = {'单身': 1, '离异': 2, '已婚': 3}
    df['marital_weight'] = df['marital_status'].map(marital_weight)
    df['family_responsibility'] = (df['age'] / 70) * 0.4 + \
                                 (df['marital_weight'] / 3) * 0.3 + \
                                 (df['family_members'] / 6) * 0.3
    
    # 6. 教育水平编码
    education_mapping = {'高中': 1, '本科': 2, '硕士': 3, '博士': 4}
    df['education_numeric'] = df['education_level'].map(education_mapping)
    
    # 7. 地区经济发展水平（简单示例，实际应根据真实数据）
    # 这里使用随机生成，实际应用中应该使用真实的经济数据
    np.random.seed(42)
    regions = df['insurance_region'].unique()
    region_development = {region: np.random.uniform(0.5, 1.0) for region in regions}
    df['region_development'] = df['insurance_region'].map(region_development)
    
    return df

def prepare_features(df, is_train=True):
    """准备特征矩阵"""
    
    # 选择特征列
    feature_columns = [
        'age', 'family_members', 'premium_amount',
        'policy_duration_days', 'premium_income_ratio',
        'occupation_stability', 'family_responsibility',
        'education_numeric', 'region_development'
    ]
    
    # 分类变量编码
    categorical_cols = ['gender', 'marital_status', 'policy_type', 'policy_term', 'claim_history']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        feature_columns.append(col + '_encoded')
    
    # 选择特征
    X = df[feature_columns]
    
    if is_train:
        y = df['renewal'].map({'Yes': 1, 'No': 0})
        return X, y
    else:
        return X

def train_random_forest(X_train, y_train):
    """训练随机森林模型"""
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 处理缺失值
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_scaled)
    
    # 训练随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # 处理类别不平衡
    )
    
    rf_model.fit(X_train_imputed, y_train)
    
    return rf_model, scaler, imputer

def main():
    print("=== 寿险客户续保预测 ===")
    
    # 1. 数据加载和预处理
    print("1. 加载和预处理数据...")
    train_df, test_df, test_ids = load_and_preprocess_data()
    
    # 2. 准备训练特征
    print("2. 准备特征...")
    X_train, y_train = prepare_features(train_df, is_train=True)
    X_test = prepare_features(test_df, is_train=False)
    
    print(f"训练特征形状: {X_train.shape}")
    print(f"测试特征形状: {X_test.shape}")
    
    # 3. 训练模型
    print("3. 训练随机森林模型...")
    model, scaler, imputer = train_random_forest(X_train, y_train)
    
    # 4. 在训练集上评估模型
    X_train_scaled = scaler.transform(X_train)
    X_train_imputed = imputer.transform(X_train_scaled)
    y_pred_train = model.predict(X_train_imputed)
    
    accuracy = accuracy_score(y_train, y_pred_train)
    print(f"训练集准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_train, y_pred_train))
    
    # 5. 对测试集进行预测
    print("4. 对测试集进行预测...")
    X_test_scaled = scaler.transform(X_test)
    X_test_imputed = imputer.transform(X_test_scaled)
    
    y_pred_test = model.predict(X_test_imputed)
    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]  # 续保概率
    
    # 6. 生成预测结果
    print("5. 生成预测结果文件...")
    results_df = pd.DataFrame({
        'policy_id': test_ids,
        'renewal': ['Yes' if pred == 1 else 'No' for pred in y_pred_test],
        'renewal_probability': y_pred_proba
    })
    
    # 保存结果到CSV文件
    results_df[['policy_id', 'renewal']].to_csv('renewal_predictions.csv', index=False, encoding='utf-8-sig')
    results_df.to_csv('renewal_predictions_with_probability.csv', index=False, encoding='utf-8-sig')
    
    print("预测完成!")
    print(f"预测结果已保存到: renewal_predictions.csv")
    print(f"包含概率的完整结果已保存到: renewal_predictions_with_probability.csv")
    
    # 显示特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== 特征重要性排名 ===")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()