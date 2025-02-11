import numpy as np
import torch
from models import DQN

# 初始化模型
state_size = 50
action_space_size = 1
model = DQN(state_size, action_space_size)
model.load_state_dict(torch.load('best_network.pth'))
model.to(device='cpu')  # 假设使用 CPU 加速预测

def get_next_state(prediction):
    # 示例预测逻辑，返回下一个状态和动作
    # 这里需要根据具体任务调整预测函数
    next_state = np.concatenate([list(row) + list(prediction)])
    return next_state, prediction[0]

# 加载测试数据
test_data = pd.read_csv('new_historical.csv')

for index, row in test_data.iterrows():
    state = np.random.randint(low=0, high=50, size=7)
    
    # 获取动作预测
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state)
        prediction = model(state_tensor)
        
        predicted_digits = prediction.cpu().numpy()
        
        # 示例误差计算逻辑（需要根据实际任务调整）
        error = abs(predicted_digits[0] - row[0])
        
        # 可视化结果
        visualize_result(row, predicted_digits)

print("Inference complete.")
