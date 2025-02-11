from rl.core import Env

class HistoricalEnv(Env):
    def __init__(self, file_path="processed_historical.csv"):
        self.data = pd.read_csv(file_path)
        self.valid_data = self.data.drop_duplicates()
        
    def reset(self):
        current_state = self._get_state(self.valid_data[-1])
        return current_state
      
    def _get_state(self, row):
        state = np.concatenate([row.to_numpy()])
        return state
      
    def step(self, action):
        # 假设当前状态是 last_row，预测下一个状态
        next_row = self.valid_data.iloc[-1]
        
        # 示例动作处理逻辑
        predicted_digits = [0] * 7  # 示例预测
        correct_count = sum([p == a for p, a in zip(predicted_digits, list(next_row))])
        
        radius = 3
        error = abs(predicted_digits[0] - next_row[0])
        reward = 2 ** (correct_count / 7) if error <= radius else -1
        
        done = False
        if sum(predicted_digits) == 7:
            done = True
        
        # 下一个状态是当前预测值和实际值的结合
        next_state = np.concatenate([list(next_row), predicted_digits])
        
        return next_state, reward, done
      
    def get_num_actions(self):
        return 1
      
    def get_state_space_size(self):
        return len(self._get_state(self.valid_data.iloc[0]))
