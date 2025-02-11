import csv
import datetime

# 读取CSV文件
input_filename = 'euro_jackpot.csv'
output_filename = 'historical.csv'

# 存储数据
date = None  # 存储日期
n_values = []  # 存 n1-n5
e_values = []  # 存 e1, e2

with open(input_filename, mode='r', newline='') as infile:
    reader = csv.reader(infile)

    with open(output_filename, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'e1', 'e2'])

        for row in reader:  
            filtered_row = [item for item in row if item.strip()]
            new_row = filtered_row[1:]
            writer.writerow(new_row)
