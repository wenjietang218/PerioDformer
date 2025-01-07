# 将txt中的mse和mae数据提取到csv表格中
# Extract the MSE and MAE data from the TXT file into a CSV table.
import csv

# 定义输入和输出文件路径
input_file = "D:/model/PerioDformer/1_6 different_noise_result.txt"
output_file = "D:/model/PerioDformer/processing_data/1_6_result.csv"

# 读取文件并提取 mse 和 mae
data = []
with open(input_file, "r") as file:
    for line in file:
        if "mse" in line and "mae" in line:
            parts = line.split(", ")
            mse = parts[0].split(":")[1]
            mae = parts[1].split(":")[1]
            data.append({"mse": mse, "mae": mae})

# 将数据写入 CSV 文件
with open(output_file, "w", newline="") as csvfile:
    fieldnames = ["mse", "mae"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入标题
    writer.writeheader()

    # 写入数据，每五行后添加两行空行
    for i, row in enumerate(data):
        writer.writerow(row)
        if (i + 1) % 5 == 0:
            writer.writerow({})
            writer.writerow({})

