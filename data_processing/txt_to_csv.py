# 对于mlp_num取1到3 w取1到5 的超参数实验数据整理
import re
import csv

# 输入文件路径
input_file = 'D:/model/PerioDformer/12_26 NetTh result.txt'
# 输出文件路径
output_file = 'D:/model/PerioDformer/processing_data/12_26 NetTh_mse.csv'

# 初始化数据存储
results = []

# 读取txt文件并提取mse值
with open(input_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if "mse:" in line:
            mse_match = re.search(r"mse:([\d.]+)", line)
            if mse_match:
                mse_value = float(mse_match.group(1))
                results.append(mse_value)

# 参数配置
block_num = 4
block_size = 15 * block_num  # 每块数据的数量
columns_per_block = 3  # 每块使用的列数
column_gap = 1  # 块之间的列间隔
rows_per_section = 5  # 每节有多少行数据
rows_per_block = (rows_per_section + 1) * block_num  # 每块需要的总行数（每节5行+1空行）*9列

# 计算输出矩阵大小
num_blocks = (len(results) + block_size - 1) // block_size  # 总块数
total_columns = num_blocks * (columns_per_block + column_gap) - column_gap  # 总列数，减去最后一个块的间隔
total_rows = rows_per_block  # 总行数

# 创建输出矩阵
output_matrix = [["" for _ in range(total_columns)] for _ in range(total_rows)]

# 填充矩阵
for block_idx in range(num_blocks):
    block_start = block_idx * block_size
    block = results[block_start:block_start + block_size]
    start_col = block_idx * (columns_per_block + column_gap)

    for idx, mse in enumerate(block):
        # 计算目标行和列
        # 每个块内的行索引，每行有3列
        row_in_section = idx // 3  # 计算当前数据在块中的行号（每块包含3列，每3个元素算作1行）
        section_with_gap = row_in_section + (row_in_section // rows_per_section)  # 加上空行
        target_row = section_with_gap

        # 计算当前数据在块内的列索引（每块3列，总共9列）
        target_col = start_col + (idx % 3)  # 当前列在块内的位置

        # 安全检查，避免越界
        if target_row >= total_rows or target_col >= total_columns:
            print(f"Warning: Skipping index ({target_row}, {target_col}) due to size constraints.")
            continue

        output_matrix[target_row][target_col] = mse

# 写入CSV文件
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output_matrix)

print(f"格式化后的MSE数据已保存到 {output_file}")
