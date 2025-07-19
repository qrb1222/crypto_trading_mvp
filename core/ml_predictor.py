import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, Tuple, List
import numba
from numba import jit, prange

class ActorCritic(nn.Module):
    """PPO的Actor-Critic网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor分支
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic分支
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim)
        self.critic_value = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """使用正交初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Actor最后一层使用较小的初始化
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 共享层
        x = torch.relu(self.shared_fc1(state))
        x = torch.relu(self.shared_fc2(x))
        
        # Actor分支
        actor_features = torch.relu(self.actor_fc(x))
        action_mean = self.actor_mean(actor_features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Critic分支
        critic_features = torch.relu(self.critic_fc(x))
        value = self.critic_value(critic_features)
        
        return action_mean, action_std, value
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成动作"""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            action = action_mean
        else:
            # 从正态分布采样
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            
        # 使用tanh限制动作范围到[-1, 1]
        action = torch.tanh(action)
        
        # 计算log概率（考虑tanh变换）
        if not deterministic:
            log_prob = dist.log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
        else:
            log_prob = torch.zeros_like(value)
        
        return action, value, log_prob

class PPOMemory:
    """PPO经验缓冲区"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
    def get_batch(self):
        return (self.states, self.actions, self.rewards, 
                self.values, self.log_probs, self.dones)

class PPOAgent:
    """完整的PPO智能体实现"""
    def __init__(self, state_dim: int, action_dim: int = 1, 
                 lr: float = 3e-4, device: str = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # PPO超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.batch_size = 64
        
        # 内存
        self.memory = PPOMemory()
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = deque(maxlen=100)
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float, float]:
        """选择动作（优化延迟）"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, value, log_prob = self.actor_critic.act(state_tensor, deterministic)
            
        return (action.cpu().numpy()[0, 0], 
                value.cpu().numpy()[0, 0], 
                log_prob.cpu().numpy()[0, 0])
    
    def store_transition(self, state: np.ndarray, action: float, reward: float, 
                        value: float, log_prob: float, done: bool):
        """存储转换"""
        self.memory.push(state, action, reward, value, log_prob, done)
        
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool], next_value: float) -> torch.Tensor:
        """计算广义优势估计(GAE)"""
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                
            advantages.insert(0, gae)
            
        return torch.FloatTensor(advantages).to(self.device)
    
    def update(self, next_value: float = 0.0):
        """PPO更新"""
        if len(self.memory.states) < self.batch_size:
            return
        
        states, actions, rewards, values, log_probs, dones = self.memory.get_batch()
        
        # 计算优势和目标值
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        # PPO训练循环
        for epoch in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # 计算当前策略的动作概率
                action_mean, action_std, values = self.actor_critic(batch_states)
                dist = torch.distributions.Normal(action_mean, action_std)
                
                # 重新计算log概率
                new_log_probs = dist.log_prob(batch_actions)
                new_log_probs = new_log_probs - torch.log(1 - batch_actions.pow(2) + 1e-6)
                new_log_probs = new_log_probs.sum(-1, keepdim=True)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算代理损失
                surr1 = ratio * batch_advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 值函数损失
                value_loss = self.value_loss_coef * nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # 熵损失
                entropy_loss = -self.entropy_coef * dist.entropy().mean()
                
                # 总损失
                loss = policy_loss + value_loss + entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
        # 清空内存
        self.memory.clear()
        self.training_step += 1
        
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

@jit(nopython=True)
def compute_technical_features_numba(prices: np.ndarray, volumes: np.ndarray, 
                                    high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """使用Numba加速的技术指标计算"""
    n = len(prices)
    features = np.zeros(30)
    
    if n < 20:
        return features
    
    # 价格变化率
    returns = np.zeros(n-1)
    for i in range(1, n):
        returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
    
    # 基础统计
    features[0] = returns[-1] if len(returns) > 0 else 0  # 最新收益率
    features[1] = np.mean(returns[-5:]) if len(returns) >= 5 else 0  # 5期均值
    features[2] = np.std(returns[-20:]) if len(returns) >= 20 else 0  # 20期波动率
    
    # RSI计算
    gains = np.zeros(len(returns))
    losses = np.zeros(len(returns))
    for i in range(len(returns)):
        if returns[i] > 0:
            gains[i] = returns[i]
        else:
            losses[i] = -returns[i]
    
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        features[3] = 100 - 100 / (1 + rs)
    else:
        features[3] = 100 if avg_gain > 0 else 50
    
    # 成交量指标
    if len(volumes) >= 20:
        features[4] = volumes[-1] / np.mean(volumes[-20:])
        features[5] = np.std(volumes[-20:]) / (np.mean(volumes[-20:]) + 1e-10)
    
    # 价格位置（布林带位置）
    if n >= 20:
        sma20 = np.mean(prices[-20:])
        std20 = np.std(prices[-20:])
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
        features[6] = (prices[-1] - lower_band) / (upper_band - lower_band + 1e-10)
    
    # MACD
    if n >= 26:
        # 简化的EMA计算
        ema12 = prices[-1]
        ema26 = prices[-1]
        alpha12 = 2.0 / 13
        alpha26 = 2.0 / 27
        
        for i in range(n-26, n):
            ema12 = alpha12 * prices[i] + (1 - alpha12) * ema12
            ema26 = alpha26 * prices[i] + (1 - alpha26) * ema26
        
        features[7] = (ema12 - ema26) / prices[-1]  # MACD标准化
    
    # ATR (Average True Range)
    if n >= 14:
        tr_values = np.zeros(14)
        for i in range(n-14, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - prices[i-1]) if i > 0 else 0
            tr3 = abs(low[i] - prices[i-1]) if i > 0 else 0
            tr_values[i-(n-14)] = max(tr1, tr2, tr3)
        
        features[8] = np.mean(tr_values) / prices[-1]  # ATR标准化
    
    # 动量指标
    for lag in [1, 5, 10]:
        if n > lag:
            features[9 + lag//5] = (prices[-1] - prices[-lag-1]) / prices[-lag-1]
    
    # 成交量加权平均价格（VWAP）
    if n >= 20:
        vwap = np.sum(prices[-20:] * volumes[-20:]) / np.sum(volumes[-20:])
        features[12] = (prices[-1] - vwap) / vwap
    
    # 更多特征填充
    return features

class LightGBMPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.ppo_agent = None  # PPO代理
        self.state_dim = 35  # 特征维度
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征 - 使用Numba加速"""
        features = df.copy()
        
        # 使用Numba加速的特征计算
        if len(df) >= 20:
            technical_features = compute_technical_features_numba(
                df['close'].values,
                df['volume'].values,
                df['high'].values,
                df['low'].values
            )
            
            # 添加计算的特征
            for i, feature_name in enumerate([
                'return_1', 'return_mean_5', 'volatility_20', 'rsi_14',
                'volume_ratio', 'volume_std', 'bb_position', 'macd_norm',
                'atr_norm', 'momentum_1', 'momentum_5', 'momentum_10', 'vwap_ratio'
            ]):
                if i < len(technical_features):
                    features[feature_name] = technical_features[i]
        
        # 添加滞后特征
        for lag in [1, 5, 10, 20]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
            
        # 添加滚动统计
        for window in [5, 10, 20]:
            features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
            features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
            features[f'volume_mean_{window}'] = features['volume'].rolling(window).mean()
            
        # 时间特征
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        
        # 删除NaN值
        features = features.dropna()
        
        # 选择特征列
        self.feature_columns = [col for col in features.columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume', 'returns']]
        
        return features[self.feature_columns]
    
    def train(self, df: pd.DataFrame, target_col: str = 'returns', 
              horizon: int = 5):
        """训练模型 - 包括LightGBM和PPO"""
        # 准备特征
        features = self.prepare_features(df)
        
        # 创建目标变量（未来收益）
        target = df['returns'].shift(-horizon)
        target = target.loc[features.index]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, shuffle=False
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练LightGBM
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        # 优化的参数以提高速度
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,  # 减少以加快推理
            'max_depth': 5,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'num_threads': 4,
            'force_col_wise': True  # 列并行
        }
        
        # 回调函数
        callbacks = [
            lgb.early_stopping(100),
            lgb.log_evaluation(0)
        ]
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=500,
            callbacks=callbacks
        )
        
        # 保存模型
        self.models[f'horizon_{horizon}'] = model
        self.scalers[f'horizon_{horizon}'] = scaler
        
        # 特征重要性
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        print(f"Top 10 features:\n{importance.head(10)}")
        
        # 训练PPO（如果还没有初始化）
        if self.ppo_agent is None:
            self.ppo_agent = PPOAgent(state_dim=len(self.feature_columns) + 5)  # +5 for additional state
        
        # 使用历史数据训练PPO
        self._train_ppo_with_history(X_train_scaled, y_train.values)
        
        return model
    
    def _train_ppo_with_history(self, features: np.ndarray, returns: np.ndarray, 
                               episodes: int = 100):
        """使用历史数据训练PPO"""
        print(f"Training PPO with {episodes} episodes...")
        
        for episode in range(episodes):
            # 随机选择起始点
            start_idx = np.random.randint(0, len(features) - 1000)
            
            total_reward = 0
            state = self._create_ppo_state(features[start_idx], 0, 0, 100000)
            
            for t in range(start_idx, min(start_idx + 1000, len(features) - 1)):
                # 选择动作
                action, value, log_prob = self.ppo_agent.select_action(state)
                
                # 计算奖励
                position = action  # -1到1之间
                actual_return = returns[t]
                reward = position * actual_return * 100  # 放大奖励信号
                
                # 风险惩罚
                if abs(position) > 0.5:
                    reward -= abs(position) * 0.1
                
                # 下一个状态
                next_state = self._create_ppo_state(
                    features[t + 1], 
                    position, 
                    reward,
                    100000 + total_reward
                )
                
                # 存储经验
                self.ppo_agent.store_transition(state, action, reward, value, log_prob, False)
                
                state = next_state
                total_reward += reward
                
                # 定期更新
                if (t - start_idx + 1) % 100 == 0:
                    self.ppo_agent.update()
            
            # Episode结束更新
            self.ppo_agent.update()
            self.ppo_agent.episode_rewards.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.ppo_agent.episode_rewards)
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
    
    def _create_ppo_state(self, features: np.ndarray, position: float, 
                         last_reward: float, equity: float) -> np.ndarray:
        """创建PPO状态向量"""
        # 添加额外的状态信息
        additional_state = np.array([
            position,  # 当前仓位
            last_reward / 100,  # 标准化的上一步奖励
            equity / 100000,  # 标准化的权益
            0,  # 可以添加更多信息
            0
        ])
        
        return np.concatenate([features, additional_state])
    
    def predict(self, df: pd.DataFrame, horizon: int = 5) -> Dict:
        """生成预测 - 结合LightGBM和PPO"""
        model_key = f'horizon_{horizon}'
        if model_key not in self.models:
            raise ValueError(f"Model for horizon {horizon} not trained")
            
        features = self.prepare_features(df)
        if features.empty:
            return {
                'predicted_return': 0,
                'direction': 'HOLD',
                'confidence': 0,
                'ppo_action': 0,
                'ppo_value': 0
            }
            
        latest_features = features.iloc[-1:].values
        
        # LightGBM预测
        scaled_features = self.scalers[model_key].transform(latest_features)
        lgb_prediction = self.models[model_key].predict(scaled_features)[0]
        
        # PPO预测
        ppo_state = self._create_ppo_state(scaled_features[0], 0, 0, 100000)
        ppo_action, ppo_value, _ = self.ppo_agent.select_action(ppo_state, deterministic=True)
        
        # 组合预测
        # 如果LightGBM和PPO方向一致，增加置信度
        lgb_direction = 'BUY' if lgb_prediction > 0.0001 else 'SELL' if lgb_prediction < -0.0001 else 'HOLD'
        ppo_direction = 'BUY' if ppo_action > 0.1 else 'SELL' if ppo_action < -0.1 else 'HOLD'
        
        if lgb_direction == ppo_direction and lgb_direction != 'HOLD':
            confidence = min(abs(lgb_prediction) * 100 + abs(ppo_action), 1.0)
            final_direction = lgb_direction
        else:
            confidence = min(abs(lgb_prediction) * 50, 0.7)
            final_direction = lgb_direction
        
        return {
            'predicted_return': lgb_prediction,
            'direction': final_direction,
            'confidence': confidence,
            'ppo_action': float(ppo_action),
            'ppo_value': float(ppo_value),
            'horizon': horizon
        }
    
    def save_models(self, directory: str):
        """保存所有模型"""
        os.makedirs(directory, exist_ok=True)
        
        # 保存LightGBM模型
        for key, model in self.models.items():
            model.save_model(os.path.join(directory, f'{key}.txt'))
        
        # 保存标准化器
        for key, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(directory, f'{key}_scaler.pkl'))
        
        # 保存PPO模型
        if self.ppo_agent is not None:
            self.ppo_agent.save(os.path.join(directory, 'ppo_model.pth'))
    
    def load_models(self, directory: str):
        """加载所有模型"""
        # 加载LightGBM模型
        for filename in os.listdir(directory):
            if filename.endswith('.txt') and filename.startswith('horizon_'):
                key = filename.replace('.txt', '')
                self.models[key] = lgb.Booster(model_file=os.path.join(directory, filename))
            elif filename.endswith('_scaler.pkl'):
                key = filename.replace('_scaler.pkl', '')
                self.scalers[key] = joblib.load(os.path.join(directory, filename))
        
        # 加载PPO模型
        ppo_path = os.path.join(directory, 'ppo_model.pth')
        if os.path.exists(ppo_path):
            if self.ppo_agent is None:
                self.ppo_agent = PPOAgent(state_dim=self.state_dim)
            self.ppo_agent.load(ppo_path)
