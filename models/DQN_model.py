import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        
        # 输入层和隐藏层
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 目标网络
        self.target = DQN(state_size, action_size).to(device)
        
    def forward(self, x):
        if device is not None:
            x = x.to(device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
  
    def act(self, state):
        q_val = self.forward(state).max()
        return torch.argmax(q_val).item()  # 返回预测的动作索引

    def target_update(self, source, target_network, target_hard=True):
        if device is not None:
            source = source.to(device)
        target_network.load_state_dict(source.state_dict())
        if target_hard:
            target_network.fc1.weight.data = source.fc1.weight.data  # 假设只更新部分参数
