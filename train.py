from models import DQN
from env import HistoricalEnv
from agents import DQN_Agent

# 初始化训练参数
state_size = 50  # 状态空间大小（每个行7个数字，总共有7个数字）
action_space_size = 1  # 动作空间大小

agent = DQN_Agent(state_size, action_space_size)
env = HistoricalEnv()

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent.to(device)

# 训练过程
best_reward = -1
num_episodes = 1000
total_episode_steps = 1000

for episode in range(num_episodes):
    reward = train_model(agent, env)  # 调用训练函数
    
    if reward > best_reward:
        best_reward = reward
        torch.save(agent.network.state_dict(), 'best_network.pth')
        torch.save(agent.target_network.state_dict(), 'best_target_network.pth')
        
    print(f"Episode {episode}: Total Reward = {reward}")

print("Training complete.")
