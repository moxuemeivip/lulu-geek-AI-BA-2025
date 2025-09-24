import pandas as pd

# 读取Excel文件
try:
    # 直接读取Excel文件，使用header=0确保第一行作为列名
    df = pd.read_excel('不同维度的数据统计.xlsx', header=0)
    
    # 打印基本信息
    print(f"文件包含 {df.shape[0]} 行，{df.shape[1]} 列")
    print("\n列名列表:\n")
    print(df.columns.tolist())
    
    # 打印完整数据（最多显示所有行，但在终端中可能仍会有显示限制）
    print("\n完整表格数据:\n")
    # 为了确保在终端中能尽可能显示完整，我们将数据转换为字符串并逐行打印
    # 首先打印表头
    header = '\t'.join(str(col) for col in df.columns)
    print(header)
    # 然后打印每行数据
    for _, row in df.iterrows():
        row_str = '\t'.join(str(val) for val in row)
        print(row_str)
    
    print("\n\n数据已完整读取并尝试显示。如果在终端中显示不完整，建议将结果导出到CSV文件查看。")
    
except Exception as e:
    print(f"读取文件时发生错误: {e}")