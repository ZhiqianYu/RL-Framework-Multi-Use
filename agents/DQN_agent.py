from rl.core import Agent, Env

class DQN_Agent(Agent):
    def __init__(self, state_size, action_size, device='cpu'):
        super().__init__()
        
        # 定义网络
        self.network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        
        # 超参数
        self.memory = ExperienceReplay(max_size=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.exploration_rate = 0.1
        self.radius = 3  # 在预测中使用的误差半径
      
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters())
        
        # 训练过程相关设置
        self.max_steps = 1000
        self.num_epochs = 100
        
    def get_action(self, state):
        if np.random.random() < self.exploration_rate:
            return random.randint(0, action_space_size - 1)
          
        with torch.no_grad():
            q_val = self.network(state).max()
            return torch.argmax(q_val).item()
      
    def remember(self, transition):
        self.memory.add(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            done=transition.done
        )
      
    def update_hard(self):
        # 硬更新代理网络和目标网络
        self.network.update_hard()
        self.target_network.update_hard()
        
    def update_soft(self):
        # 软更新目标网络
        target_update_loss = 0.0
        for t in self.memory.sample(self.batch_size):
            state = torch.FloatTensor(t.state)
            action = torch.LongTensor([t.action])
            reward = torch.FloatTensor([t.reward])
            next_state = torch.FloatTensor(t.next_state)
            
            optimizer.zero_grad()
            q_target = loss_fn(self.target_network(state), next_state)
            q_pred = self.network(state).gather(1, action.unsqueeze(1))
            
            # 分开计算价值函数和策略函数的损失
            target_update_loss = F.mse_loss(q_target[:, 0], q_pred[:, 0])
            loss = F.mse_loss(q_pred[:, 1:], q_target[:, 1:])
            
            optimizer.backward(loss)
            optimizer.step()
        
        # 平滑地更新目标网络
        for param in self.target_network.parameters():
            param.data = (param.data * 0.01) + (param.grad * 0.99)
