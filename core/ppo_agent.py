import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, Tuple, List

class ActorCritic(nn.Module):
    """PPO的Actor-Critic网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor头
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic头
        self.critic = nn.Linear(hidden_dim, 1)
        
        # 初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        shared_features = self.shared(state)
        
        # Actor输出
        action_mean = self.actor_mean(shared_features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Critic输出
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def act(self, state, deterministic=False):
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            action = action_mean
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            
        # 限制动作范围 [-1, 1]
        action = torch.tanh(action)
        
        return action, value

class PPOAgent:
    """高性能PPO实现，针对高频交易优化"""
    def __init__(self, state_dim: int, action_dim: int = 1, lr: float = 3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # PPO超参数
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float]:
        """快速推理，优化延迟"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value = self.actor_critic.act(state_tensor, deterministic)
            
        return action.cpu().numpy()[0, 0], value.cpu().numpy()[0, 0]
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(1 - done)
    
    def compute_gae(self, next_value):
        """计算广义优势估计(GAE)"""
        values = self.values + [next_value]
        gae = 0
        advantages = []
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * self.masks[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * self.masks[t] * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def update(self, next_value):
        """PPO更新"""
        # 计算优势和回报
        advantages = self.compute_gae(next_value)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.values).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # 多轮更新
        for _ in range(10):  # PPO epochs
            # 前向传播
            action_mean, action_std, values = self.actor_critic(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            # 计算新的log概率
            new_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算surrogate损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 值函数损失
            value_loss = self.value_loss_coef * nn.MSELoss()(values.squeeze(), returns)
            
            # 熵正则化
            entropy = dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy
            
            # 总损失
            loss = policy_loss + value_loss + entropy_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # 清空缓冲区
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.masks.clear()
