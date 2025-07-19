import numpy as np
from typing import Dict, List
from numba import jit

class EnhancedRiskManager:
    """针对高胜率优化的风险管理系统"""
    
    def __init__(self):
        self.max_leverage = 30
        self.max_drawdown = 0.4
        self.target_win_rate = 0.75  # 目标胜率75%
        
        # 动态风险参数
        self.confidence_threshold = 0.75
        self.kelly_multiplier = 0.2  # 更保守的Kelly
        self.max_correlation = 0.6   # 降低相关性阈值
        
        # 性能追踪
        self.trade_history = []
        self.rolling_win_rate = 0.5
        self.rolling_sharpe = 0.0
        
    def calculate_optimal_position_size(self, signal: Dict, market_state: Dict) -> float:
        """计算最优仓位大小，优化胜率"""
        
        # 基础Kelly计算
        win_rate = self._estimate_win_probability(signal)
        avg_win = self._estimate_avg_win(market_state)
        avg_loss = self._estimate_avg_loss(market_state)
        
        # 修正的Kelly公式
        if avg_win > 0 and win_rate > 0.5:
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly = max(0, min(kelly, 0.25))  # 限制最大Kelly
        else:
            kelly = 0
        
        # 多层调整
        size_multipliers = {
            'kelly': kelly,
            'confidence': min(signal['quality_score'] / self.confidence_threshold, 1.0),
            'volatility': self._volatility_adjustment(market_state),
            'regime': self._regime_adjustment(market_state),
            'drawdown': self._drawdown_adjustment(),
            'correlation': self._correlation_adjustment(signal)
        }
        
        # 计算最终仓位
        base_size = 0.02  # 基础2%仓位
        final_size = base_size * np.prod(list(size_multipliers.values()))
        
        # 严格限制
        final_size = min(final_size, 0.1)  # 最大10%
        final_size = max(final_size, 0.001)  # 最小0.1%
        
        return final_size
    
    def _estimate_win_probability(self, signal: Dict) -> float:
        """估计获胜概率"""
        # 基于多个因素估计
        ml_confidence = signal['ml_confidence']
        ppo_confidence = abs(signal['ppo_action'])
        quality_score = signal['quality_score']
        
        # 加权平均
        base_win_rate = 0.5 + 0.2 * ml_confidence + 0.15 * ppo_confidence + 0.15 * quality_score
        
        # 根据历史表现调整
        history_adjustment = 0.7 * self.rolling_win_rate + 0.3 * base_win_rate
        
        return min(history_adjustment, 0.85)  # 上限85%
    
    def _volatility_adjustment(self, market_state: Dict) -> float:
        """波动率调整系数"""
        current_vol = market_state.get('volatility', 0.02)
        
        # 反向调整：波动率越高，仓位越小
        if current_vol < 0.01:
            return 1.2
        elif current_vol < 0.02:
            return 1.0
        elif current_vol < 0.03:
            return 0.7
        else:
            return 0.4
    
    def _regime_adjustment(self, market_state: Dict) -> float:
        """市场状态调整"""
        regime = market_state.get('regime', 'unknown')
        
        adjustments = {
            'trending': 1.2,
            'ranging': 0.8,
            'volatile': 0.5,
            'unknown': 0.7
        }
        
        return adjustments.get(regime, 0.7)
    
    def _drawdown_adjustment(self) -> float:
        """回撤调整"""
        current_dd = self._calculate_current_drawdown()
        
        if current_dd < 0.1:
            return 1.0
        elif current_dd < 0.2:
            return 0.7
        elif current_dd < 0.3:
            return 0.4
        else:
            return 0.2  # 大幅降低仓位
    
    def _correlation_adjustment(self, signal: Dict) -> float:
        """相关性调整，避免同向押注"""
        # 检查与现有仓位的相关性
        # 这里简化处理
        return 0.8
    
    def should_take_trade(self, signal: Dict, market_state: Dict) -> bool:
        """决定是否执行交易"""
        
        # 多重检查提高胜率
        checks = {
            'quality_score': signal['quality_score'] >= self.confidence_threshold,
            'ml_confidence': signal['ml_confidence'] >= 0.7,
            'ppo_alignment': abs(signal['ppo_action']) >= 0.3,
            'not_anomaly': market_state.get('anomaly_score', 0) < 0.5,
            'spread_ok': market_state['spread'] < market_state['avg_spread'] * 2,
            'volume_ok': market_state['volume'] > market_state['avg_volume'] * 0.3,
            'drawdown_ok': self._calculate_current_drawdown() < 0.35,
            'correlation_ok': self._check_correlation_limit(signal)
        }
        
        # 所有检查必须通过
        return all(checks.values())
    
    def _calculate_current_drawdown(self) -> float:
        """计算当前回撤"""
        if not self.trade_history:
            return 0.0
            
        equity_curve = np.cumsum([t['pnl'] for t in self.trade_history])
        if len(equity_curve) == 0:
            return 0.0
            
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / (peak + 1e-10)
        
        return float(np.max(drawdown))
    
    def _check_correlation_limit(self, signal: Dict) -> bool:
        """检查相关性限制"""
        # 简化版本，实际应该计算与现有仓位的相关性
        return True
    
    def update_performance(self, trade_result: Dict):
        """更新性能指标"""
        self.trade_history.append(trade_result)
        
        # 更新滚动胜率（最近100笔）
        recent_trades = self.trade_history[-100:]
        wins = sum(1 for t in recent_trades if t['pnl'] > 0)
        self.rolling_win_rate = wins / len(recent_trades) if recent_trades else 0.5
        
        # 更新滚动夏普比率
        if len(recent_trades) >= 20:
            returns = [t['pnl'] / t['size'] for t in recent_trades]
            self.rolling_sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
