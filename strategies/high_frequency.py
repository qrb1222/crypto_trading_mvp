import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from core.indicators import TechnicalIndicators
from core.ml_predictor import LightGBMPredictor
from core.risk_manager import RiskManager
import time
from scipy import stats
from sklearn.preprocessing import RobustScaler

class MarketRegimeDetector:
    """市场状态检测器"""
    def __init__(self, lookback_periods: int = 60):
        self.lookback_periods = lookback_periods
        self.regime_history = []
        
    def detect(self, price_data: pd.Series, volume_data: pd.Series) -> Dict:
        """检测当前市场状态"""
        if len(price_data) < self.lookback_periods:
            return {'regime': 'unknown', 'confidence': 0}
        
        # 计算收益率
        returns = price_data.pct_change().dropna()
        recent_returns = returns.tail(self.lookback_periods)
        
        # 计算各种指标
        metrics = {
            'mean_return': recent_returns.mean(),
            'volatility': recent_returns.std(),
            'skewness': stats.skew(recent_returns),
            'kurtosis': stats.kurtosis(recent_returns),
            'trend_strength': self._calculate_trend_strength(price_data),
            'volume_trend': self._calculate_volume_trend(volume_data)
        }
        
        # 判断市场状态
        regime, confidence = self._classify_regime(metrics)
        
        return {
            'regime': regime,
            'confidence': confidence,
            'metrics': metrics
        }
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """计算趋势强度"""
        if len(prices) < 20:
            return 0
        
        # 使用线性回归斜率
        x = np.arange(len(prices.tail(20)))
        y = prices.tail(20).values
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        # 标准化斜率
        normalized_slope = slope / (prices.mean() + 1e-10)
        trend_strength = abs(r_value) * np.sign(normalized_slope)
        
        return trend_strength
    
    def _calculate_volume_trend(self, volumes: pd.Series) -> float:
        """计算成交量趋势"""
        if len(volumes) < 20:
            return 1.0
        
        recent_avg = volumes.tail(5).mean()
        longer_avg = volumes.tail(20).mean()
        
        return recent_avg / (longer_avg + 1e-10)
    
    def _classify_regime(self, metrics: Dict) -> Tuple[str, float]:
        """分类市场状态"""
        volatility = metrics['volatility']
        trend_strength = metrics['trend_strength']
        volume_trend = metrics['volume_trend']
        
        # 定义阈值
        high_vol_threshold = 0.03
        trend_threshold = 0.5
        
        if volatility > high_vol_threshold:
            regime = 'volatile'
            confidence = min(volatility / high_vol_threshold, 1.0)
        elif abs(trend_strength) > trend_threshold:
            regime = 'trending'
            confidence = min(abs(trend_strength) / trend_threshold, 1.0)
        else:
            regime = 'ranging'
            confidence = 1.0 - abs(trend_strength) / trend_threshold
        
        # 成交量确认
        if volume_trend > 1.5 and regime == 'trending':
            confidence = min(confidence * 1.2, 1.0)
        elif volume_trend < 0.7:
            confidence *= 0.8
        
        return regime, confidence

class OrderFlowAnalyzer:
    """订单流分析器"""
    def __init__(self):
        self.flow_history = []
        self.toxicity_threshold = 0.001
        
    def analyze(self, order_book: Dict, recent_trades: pd.DataFrame) -> Dict:
        """分析订单流质量"""
        analysis = {
            'spread_quality': self._analyze_spread(order_book),
            'depth_imbalance': self._analyze_depth_imbalance(order_book),
            'trade_imbalance': self._analyze_trade_imbalance(recent_trades),
            'kyle_lambda': self._calculate_kyle_lambda(recent_trades),
            'is_toxic': False,
            'flow_score': 0
        }
        
        # 计算综合流动性分数
        analysis['flow_score'] = self._calculate_flow_score(analysis)
        
        # 判断是否为毒性流
        analysis['is_toxic'] = analysis['kyle_lambda'] > self.toxicity_threshold
        
        return analysis
    
    def _analyze_spread(self, order_book: Dict) -> float:
        """分析买卖价差质量"""
        if not order_book or 'spread' not in order_book:
            return 0
        
        spread = order_book['spread']
        mid_price = order_book.get('mid_price', 1)
        
        # 相对价差
        relative_spread = spread / (mid_price + 1e-10)
        
        # 价差越小质量越高
        quality = 1 / (1 + relative_spread * 10000)
        
        return quality
    
    def _analyze_depth_imbalance(self, order_book: Dict) -> float:
        """分析深度不平衡"""
        if not order_book:
            return 0
        
        bid_volume = order_book.get('bid_volume', 0)
        ask_volume = order_book.get('ask_volume', 0)
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
        
        # 计算不平衡度
        imbalance = (bid_volume - ask_volume) / total_volume
        
        return imbalance
    
    def _analyze_trade_imbalance(self, trades: pd.DataFrame) -> float:
        """分析成交不平衡"""
        if trades.empty:
            return 0
        
        buy_volume = trades[trades['side'] == 'buy']['amount'].sum()
        sell_volume = trades[trades['side'] == 'sell']['amount'].sum()
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0
        
        imbalance = (buy_volume - sell_volume) / total_volume
        
        return imbalance
    
    def _calculate_kyle_lambda(self, trades: pd.DataFrame) -> float:
        """计算Kyle's Lambda（价格冲击系数）"""
        if len(trades) < 10:
            return 0
        
        # 计算价格变化和成交量
        price_changes = trades['price'].diff().abs()
        volumes = trades['amount']
        
        # 过滤无效值
        valid_mask = (price_changes > 0) & (volumes > 0)
        price_changes = price_changes[valid_mask]
        volumes = volumes[valid_mask]
        
        if len(volumes) == 0:
            return 0
        
        # Kyle's Lambda = price_impact / sqrt(volume)
        kyle_lambda = np.sum(price_changes) / (np.sqrt(np.sum(volumes)) + 1e-10)
        
        return kyle_lambda
    
    def _calculate_flow_score(self, analysis: Dict) -> float:
        """计算综合流动性分数"""
        weights = {
            'spread_quality': 0.3,
            'depth_imbalance': 0.2,
            'trade_imbalance': 0.2,
            'kyle_lambda': 0.3
        }
        
        # 标准化各指标
        spread_score = analysis['spread_quality']
        depth_score = 1 - abs(analysis['depth_imbalance'])
        trade_score = 1 - abs(analysis['trade_imbalance'])
        kyle_score = 1 / (1 + analysis['kyle_lambda'] * 1000)
        
        # 加权平均
        flow_score = (
            weights['spread_quality'] * spread_score +
            weights['depth_imbalance'] * depth_score +
            weights['trade_imbalance'] * trade_score +
            weights['kyle_lambda'] * kyle_score
        )
        
        return flow_score

class HighFrequencyStrategy:
    """增强的高频交易策略"""
    def __init__(self, risk_manager: RiskManager, ml_predictor: LightGBMPredictor):
        self.risk_manager = risk_manager
        self.ml_predictor = ml_predictor
        
        # 策略参数
        self.min_confidence = 0.75
        self.max_positions = 3
        self.position_timeout = 300  # 5分钟超时
        
        # 分析器
        self.regime_detector = MarketRegimeDetector()
        self.flow_analyzer = OrderFlowAnalyzer()
        
        # 信号缓存和历史
        self.signal_cache = {}
        self.position_history = []
        self.active_positions = {}
        
        # 性能追踪
        self.signal_count = 0
        self.trade_count = 0
        self.win_count = 0
        
    def calculate_signal_quality(self, ml_signal: Dict, market_data: Dict, 
                               technical_indicators: Dict, order_flow: Dict,
                               market_regime: Dict) -> float:
        """计算综合信号质量分数"""
        # 基础分数
        scores = {
            'ml_confidence': ml_signal['confidence'] * 0.25,
            'ml_ppo_alignment': self._check_ml_ppo_alignment(ml_signal) * 0.20,
            'technical_alignment': self._check_technical_alignment(technical_indicators, ml_signal) * 0.15,
            'market_regime': self._evaluate_regime_fit(market_regime, ml_signal) * 0.15,
            'order_flow': order_flow['flow_score'] * 0.15,
            'volatility_fit': self._check_volatility_fit(market_data) * 0.10
        }
        
        # 计算总分
        total_score = sum(scores.values())
        
        # 惩罚因素
        if order_flow['is_toxic']:
            total_score *= 0.5
        
        if market_regime['regime'] == 'volatile' and market_regime['confidence'] > 0.8:
            total_score *= 0.7
        
        return min(total_score, 1.0)
    
    def _check_ml_ppo_alignment(self, ml_signal: Dict) -> float:
        """检查ML和PPO的一致性"""
        ml_direction = 1 if ml_signal['direction'] == 'BUY' else -1 if ml_signal['direction'] == 'SELL' else 0
        ppo_direction = np.sign(ml_signal.get('ppo_action', 0))
        
        if ml_direction == 0 or ppo_direction == 0:
            return 0
        
        if ml_direction == ppo_direction:
            # 方向一致，根据强度计算分数
            alignment = min(abs(ml_signal['ppo_action']), 1.0)
            return alignment
        else:
            # 方向不一致
            return 0
    
    def _check_technical_alignment(self, indicators: Dict, ml_signal: Dict) -> float:
        """检查技术指标对齐度"""
        alignments = []
        signal_direction = ml_signal['direction']
        
        # RSI对齐
        rsi = indicators.get('rsi', 50)
        if signal_direction == 'BUY' and rsi < 30:
            alignments.append(1.0)
        elif signal_direction == 'SELL' and rsi > 70:
            alignments.append(1.0)
        elif 40 < rsi < 60:
            alignments.append(0.5)
        else:
            alignments.append(0.2)
        
        # MACD对齐
        macd_hist = indicators.get('macd_hist', 0)
        if (signal_direction == 'BUY' and macd_hist > 0) or (signal_direction == 'SELL' and macd_hist < 0):
            alignments.append(1.0)
        else:
            alignments.append(0.3)
        
        # 布林带位置
        bb_position = indicators.get('bb_position', 0.5)
        if signal_direction == 'BUY' and bb_position < 0.2:
            alignments.append(1.0)
        elif signal_direction == 'SELL' and bb_position > 0.8:
            alignments.append(1.0)
        else:
            alignments.append(0.5)
        
        # 成交量确认
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            alignments.append(1.0)
        elif volume_ratio > 1.0:
            alignments.append(0.7)
        else:
            alignments.append(0.4)
        
        return np.mean(alignments)
    
    def _evaluate_regime_fit(self, market_regime: Dict, ml_signal: Dict) -> float:
        """评估市场状态适合度"""
        regime = market_regime['regime']
        confidence = market_regime['confidence']
        signal_direction = ml_signal['direction']
        
        # 趋势市场最适合趋势跟随
        if regime == 'trending':
            # 检查是否顺势
            trend_direction = 'BUY' if market_regime['metrics']['mean_return'] > 0 else 'SELL'
            if signal_direction == trend_direction:
                return confidence
            else:
                return confidence * 0.3
        
        # 震荡市场适合均值回归
        elif regime == 'ranging':
            return confidence * 0.7
        
        # 高波动市场需要谨慎
        elif regime == 'volatile':
            return (1 - confidence) * 0.5
        
        else:
            return 0.5
    
    def _check_volatility_fit(self, market_data: Dict) -> float:
        """检查波动率适合度"""
        volatility = market_data.get('volatility', 0.02)
        
        # 理想波动率范围
        if 0.01 < volatility < 0.025:
            return 1.0
        elif 0.005 < volatility < 0.04:
            return 0.7
        else:
            return 0.3
    
    def detect_optimal_entry(self, market_data: Dict, order_book: Dict) -> Dict:
        """检测最优入场时机"""
        # 检查价差
        spread_ratio = order_book['spread'] / order_book.get('mid_price', 1)
        
        # 检查深度
        depth_quality = min(order_book.get('bid_volume', 0) + order_book.get('ask_volume', 0), 1000) / 1000
        
        # 检查短期动量
        recent_returns = market_data.get('returns_60min', [])
        if len(recent_returns) > 5:
            micro_momentum = np.mean(recent_returns[-5:])
        else:
            micro_momentum = 0
        
        # 计算入场质量
        entry_quality = (
            (1 - spread_ratio * 10000) * 0.4 +  # 价差越小越好
            depth_quality * 0.3 +  # 深度越大越好
            (1 - abs(micro_momentum) * 100) * 0.3  # 动量不要太强
        )
        
        return {
            'quality': max(0, min(entry_quality, 1)),
            'spread_ok': spread_ratio < 0.0001,
            'depth_ok': depth_quality > 0.5,
            'momentum_ok': abs(micro_momentum) < 0.001
        }
    
    def generate_signal(self, market_data: pd.DataFrame, 
                       order_book: Dict, recent_trades: pd.DataFrame) -> Optional[Dict]:
        """生成交易信号 - 完整实现"""
        start_time = time.time()
        self.signal_count += 1
        
        # 检查是否有足够的数据
        if len(market_data) < 100:
            return None
        
        # 1. 计算技术指标
        indicators = TechnicalIndicators.calculate_features(market_data)
        latest_indicators = indicators.iloc[-1].to_dict()
        
        # 2. ML预测（包含PPO）
        try:
            ml_signal = self.ml_predictor.predict(market_data)
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None
        
        # 3. 市场状态检测
        market_regime = self.regime_detector.detect(
            market_data['close'],
            market_data['volume']
        )
        
        # 4. 订单流分析
        order_flow = self.flow_analyzer.analyze(order_book, recent_trades)
        
        # 5. 检查入场时机
        entry_timing = self.detect_optimal_entry(
            {
                'returns_60min': market_data['close'].pct_change().tail(60).values,
                'volatility': latest_indicators.get('volatility', 0.02)
            },
            order_book
        )
        
        # 6. 计算信号质量
        signal_quality = self.calculate_signal_quality(
            ml_signal,
            market_data.to_dict(),
            latest_indicators,
            order_flow,
            market_regime
        )
        
        # 7. 应用严格的过滤条件
        if signal_quality < self.min_confidence:
            return None
        
        if not entry_timing['spread_ok'] or not entry_timing['depth_ok']:
            return None
        
        if order_flow['is_toxic']:
            return None
        
        # 8. 检查现有持仓
        if len(self.active_positions) >= self.max_positions:
            return None
        
        # 9. 生成最终信号
        signal = {
            'symbol': 'BTCUSDT',  # 应该从参数传入
            'side': ml_signal['direction'],
            'ml_confidence': ml_signal['confidence'],
            'ppo_action': ml_signal['ppo_action'],
            'ppo_value': ml_signal['ppo_value'],
            'predicted_return': ml_signal['predicted_return'],
            'signal_quality': signal_quality,
            'market_regime': market_regime['regime'],
            'order_flow_score': order_flow['flow_score'],
            'entry_timing_score': entry_timing['quality'],
            'price': market_data['close'].iloc[-1],
            'timestamp': pd.Timestamp.now(),
            'generation_time': time.time() - start_time,
            'indicators': {
                'rsi': latest_indicators.get('rsi'),
                'macd': latest_indicators.get('macd'),
                'bb_position': latest_indicators.get('bb_position'),
                'volatility': latest_indicators.get('volatility')
            }
        }
        
        # 缓存信号
        self.signal_cache[signal['symbol']] = signal
        
        return signal
    
    def manage_positions(self, current_price: float) -> List[Dict]:
        """管理现有持仓"""
        actions = []
        current_time = time.time()
        
        for symbol, position in list(self.active_positions.items()):
            # 计算持仓时间
            hold_time = current_time - position['entry_time']
            
            # 计算浮动盈亏
            if position['side'] == 'BUY':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            # 止盈止损逻辑
            should_close = False
            reason = ''
            
            # 止损：-2%
            if pnl_pct < -0.02:
                should_close = True
                reason = 'stop_loss'
            
            # 止盈：根据市场状态动态调整
            elif pnl_pct > 0.01 and position.get('market_regime') == 'trending':
                # 趋势市场，使用追踪止盈
                if pnl_pct < position.get('max_pnl', 0) * 0.8:
                    should_close = True
                    reason = 'trailing_stop'
            elif pnl_pct > 0.005:
                # 非趋势市场，快速获利了结
                should_close = True
                reason = 'take_profit'
            
            # 超时平仓
            elif hold_time > self.position_timeout:
                should_close = True
                reason = 'timeout'
            
            # 更新最大盈利
            position['max_pnl'] = max(position.get('max_pnl', 0), pnl_pct)
            
            if should_close:
                action = {
                    'action': 'close',
                    'symbol': symbol,
                    'reason': reason,
                    'pnl_pct': pnl_pct,
                    'hold_time': hold_time
                }
                actions.append(action)
                
                # 更新统计
                if pnl_pct > 0:
                    self.win_count += 1
                
                # 移除持仓
                del self.active_positions[symbol]
        
        return actions
    
    def update_position(self, signal: Dict, execution_result: Dict):
        """更新持仓记录"""
        if execution_result['status'] == 'filled':
            self.active_positions[signal['symbol']] = {
                'side': signal['side'],
                'entry_price': execution_result['price'],
                'entry_time': time.time(),
                'size': execution_result['size'],
                'market_regime': signal['market_regime'],
                'signal_quality': signal['signal_quality'],
                'max_pnl': 0
            }
            self.trade_count += 1
    
    def get_performance_stats(self) -> Dict:
        """获取策略性能统计"""
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0
        signal_efficiency = self.trade_count / self.signal_count if self.signal_count > 0 else 0
        
        return {
            'total_signals': self.signal_count,
            'total_trades': self.trade_count,
            'win_trades': self.win_count,
            'win_rate': win_rate,
            'signal_efficiency': signal_efficiency,
            'active_positions': len(self.active_positions)
        }
