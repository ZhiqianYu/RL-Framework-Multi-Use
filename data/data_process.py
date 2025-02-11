import pandas as pd
import numpy as np

def process_data(input_path):
    # 加载原始数据
    df = pd.read_csv(input_path)
    
    # 数据清洗，去重和处理缺失值
    df = df.drop_duplicates()
    df = df.fillna(0)  # 假设缺失值用0填充
    
    # 分割训练集和测试集（比例可调整）
    train_df = df.iloc[:-100]
    test_df = df.iloc[-100:]
    
    return train_df, test_df

def normalize_data(df):
    # 标准化数据，假设每列均值为0
    return (df - df.mean()).values

def get_date_list():
    # 获取日期列表，可用于生成新的测试数据
    current_date = datetime.today().date()
    date_list = [current_date - timedelta(days=i) for i in range(7)]
    return date_list

if __name__ == "__main__":
    input_path = "processed_historical.csv"
    train_df, test_df = process_data(input_path)
    normalized_train = normalize_data(train_df)
    print("Data loaded and processed.")
