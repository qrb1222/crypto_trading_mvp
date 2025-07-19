import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from core.fast_predictor import FastLightGBMPredictor
from core.ppo_agent import PPOAgent
import asyncio
from concurrent.futures import ThreadPoolExecutor

class EnhancedHighFrequencyStrategy:
    """生产级高频交易策略，集成PPO和优化的ML"""
    
    def __init__(self):
        self.ml_predictor = FastLightGBMPredictor()
        self.ppo_agent = PPOAgent(state_dim=30)  # 假设30个特征
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 策略参数
        self.min_confidence = 0.75  # 提高到75%以增加胜率
        self.max_positions = 3      # 限制同时持仓
        self.signal_cache = {}      # 信号缓存
        
        # 高级过滤器
        self.regime_detector = MarketRegimeDetector()
        self.anomaly_detector = AnomalyDetector()
        
    def calculate_enhanced_signal_quality(self, ml_signal: Dict, 
                                        ppo_action: float,
                                        market_state: Dict) -> float:
        """增强的信号质量评分系统"""
        
        # 基础ML信号强度
        ml_score = ml_signal['confidence'] * 0.25
        
        # PPO动作确认
        ppo_alignment = 1.0 if (ppo_action > 0) == (ml_signal['direction'] == 'BUY') else 0.0
        ppo_score = ppo_alignment * abs(ppo_action) * 0.25
        
        # 市场状态评分
        regime_score = self._evaluate_market_regime(market_state) * 0.2
        
        # 异常检测评分（反向指标）
        anomaly_score = (1 - self.anomaly_detector.detect(market_state)) * 0.15
        
        # 订单流质量
        flow_score = self._evaluate_order_flow(market_state) * 0.15
        
        total_score = ml_score + ppo_score + regime_score + anomaly_score + flow_score
        
        return total_score
    
    def _evaluate_market_regime(self, market_state: Dict) -> float:
        """评估市场状态适合度"""
        regime = self.regime_detector.detect(market_state)
        
        # 趋势市场最适合
        if regime == 'trending':
            return 1.0
        elif regime == 'ranging':
            return 0.5
        else:  # volatile
            return 0.3
    
    def _evaluate_order_flow(self, market_state: Dict) -> float:
        """评估订单流质量"""
        spread_ratio = market_state['spread'] / market_state['avg_spread']
        imbalance = abs(market_state['imbalance'])
        
        # 低点差 + 高不平衡 = 好信号
        spread_score = 1 / (1 + spread_ratio)
        imbalance_score = min(imbalance * 2, 1.0)
        
        return (spread_score + imbalance_score) / 2
    
    async def generate_ultra_fast_signal(self, market_data: Dict) -> Optional[Dict]:
        """超快速信号生成（目标<100微秒）"""
        
        # 并行执行预测
        features = self.ml_predictor.prepare_features_fast(market_data)
        
        # 异步执行ML和PPO预测
        ml_future = self.executor.submit(self.ml_predictor.predict_ultra_fast, features)
        
        # PPO需要更多状态信息
        state = self._prepare_ppo_state(market_data, features)
        ppo_action, ppo_value = self.ppo_agent.act(state, deterministic=False)
        
        # 获取ML结果
        ml_signal = ml_future.result()
        
        # 快速质量检查
        if ml_signal['confidence'] < 0.6:  # 第一层过滤
            return None
            
        # 计算综合信号质量
        signal_quality = self.calculate_enhanced_signal_quality(
            ml_signal, ppo_action, market_data
        )
        
        # 严格的质量阈值
        if signal_quality < self.min_confidence:
            return None
        
        # 生成最终信号
        signal = {
            'symbol': market_data['symbol'],
            'direction': 'BUY' if ppo_action > 0 else 'SELL',
            'strength': abs(ppo_action),
            'ml_confidence': ml_signal['confidence'],
            'ppo_action': ppo_action,
            'ppo_value': ppo_value,
            'quality_score': signal_quality,
            'predicted_return': ml_signal['predicted_return'],
            'timestamp': pd.Timestamp.now()
        }
        
        return signal
    
    def _prepare_ppo_state(self, market_data: Dict, features: np.ndarray) -> np.ndarray:
        """准备PPO状态向量"""
        # 结合特征和额外的市场状态
        position_info = np.array([
            market_data.get('current_position', 0),
            market_data.get('unrealized_pnl', 0),
            market_data.get('total_equity', 100000) / 100000  # 归一化
        ])
        
        return np.concatenate([features, position_info])

class MarketRegimeDetector:
    """快速市场状态检测"""
    
    def __init__(self):
        self.lookback = 60  # 60分钟
        self.regimes = ['trending', 'ranging', 'volatile']
        
    def detect(self, market_state: Dict) -> str:
        """检测当前市场状态"""
        returns = market_state.get('returns_60min', [])
        
        if len(returns) < self.lookback:
            return 'unknown'
        
        # 计算趋势强度
        trend_strength = abs(np.mean(returns)) / (np.std(returns) + 1e-10)
        
        # 计算波动率
        volatility = np.std(returns)
        
        if trend_strength > 0.5:
            return 'trending'
        elif volatility > 0.02:
            return 'volatile'
        else:
            return 'ranging'

class AnomalyDetector:
    """快速异常检测"""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        
    def detect(self, market_state: Dict) -> float:
        """返回异常概率[0,1]"""
        # Z-score异常检测
        current_return = market_state.get('last_return', 0)
        mean_return = market_state.get('mean_return_60min', 0)
        std_return = market_state.get('std_return_60min', 0.01)
        
        z_score = abs(current_return - mean_return) / std_return
        
        # 检查多个异常指标
        anomaly_scores = [
            min(z_score / self.threshold, 1.0),  # 价格异常
            1.0 if market_state.get('spread', 0) > market_state.get('avg_spread', 1) * 3 else 0.0,  # 点差异常
            1.0 if market_state.get('volume', 0) < market_state.get('avg_volume', 1) * 0.1 else 0.0  # 成交量异常
        ]
        
        return max(anomaly_scores)
