import numpy as np
import pandas as pd
from collections import deque

# 经验回放缓存
class ExperienceReplay:
    def __init__(self, max_size=10000):
        self.memory = deque(maxsize=max_size)
        
    def remember(self, transition):
        self.memory.append(transition)
        
    def sample(self, batch_size=64):
        return random.sample(self.memory, batch_size)
        
    def update_hard(self):
        # 硬更新所有记忆到当前网络
        for t in self.memory:
            state = torch.FloatTensor(t.state)
            action = torch.LongTensor([t.action])
            reward = torch.FloatTensor([t.reward])
            next_state = torch.FloatTensor(t.next_state)
            
            # 计算 Q_target 和 Q_pred
            optimizer.zero_grad()
            q_target = loss_fn(agent.target_network(state), next_state)
            q_pred = agent.network(state).gather(1, action.unsqueeze(1))
            
            # 更新目标网络（只更新值函数部分）
            target_update_loss = F.mse_loss(q_target, q_pred[:, 0])
            agent.target_network.fc1.weight.grad = torch.zeros_like(agent.target_network.fc1.weight)
            agent.target_network.fc1.weight.data = (agent.target_network.fc1.weight.data * 0.01) + (state.weight * 0.99)
            
            # 更新网络
            loss = target_update_loss + F.mse_loss(q_pred[:, 1:], q_target[:, 1:])
            agent.network.backward(loss)
            optimizer.step()

def calculate_accuracy(predicted, actual):
    return np.mean(np.sum(predicted == actual, axis=0))

# 检查是否有可用的GPU
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

# 获取模型并加载到设备上
def move_model_to_device(model, device):
    model.to(device)
    return model

# 检查模型是否在正确的设备上
def is_on_device(model, device):
    return next(model.parameters()).device == device.type_as(model)

