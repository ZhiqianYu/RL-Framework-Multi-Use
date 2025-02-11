import torch
import torch.optim as optim
from rl.core import Env

def train_model(agent, env, num_epochs=100, batch_size=64):
    optimizer = optim.Adam(agent.parameters())
    loss_fn = nn.MSELoss()
    
    for epoch in range(num_epochs):
        avg_loss = 0.0
        total_steps = 0
        
        # Experience replay buffer更新
        with torch.no_grad():
            current_state = env.reset()
            action = agent.get_action(current_state)
            next_state, reward, done = env.step(action)
            
            if done:
                break
              
            # 将经验转换为 tensors
            exp = Transition(current_state, action, reward, None, done)
            agent.remember(transition=exp)
            
            # 采样小批量数据
            while len(agent.memory) > batch_size:
                mini_batch = random.sample(agent.memory, batch_size)
                states = torch.FloatTensor(np.array([t.state for t in mini_batch]))
                actions = torch.LongTensor([t.action for t in mini_batch])
                rewards = torch.FloatTensor([t.reward for t in mini_batch])
                next_states = torch.FloatTensor(np.array([t.next_state for t in mini_batch]))
                
                # 计算目标值：Q_target - Q_pred
                optimizer.zero_grad()
                q_target = loss_fn(agent.target_network(states), next_states)
                q_pred = agent.network(states).gather(1, actions.unsqueeze(1))
                loss = F.mse_loss(q_pred[:, 0], q_target[:, 0])
                
                # 更新网络
                agent.network.backward(loss)
                optimizer.step()
                
            if len(agent.memory) >= batch_size:
                agent.update_hard()
                
        print(f"Epoch {epoch} | Loss: {loss.item()}")
    return agent

# 初始化代理和环境
state_size = env.get_state_space_size()
action_size = env.get_num_actions()
agent = DQN(state_size, action_size)
trained_agent = train_model(agent, env)
