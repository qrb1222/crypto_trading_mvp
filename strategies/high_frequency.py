import pandas as pd
import numpy as np
from typing import Dict, Optional
from core.indicators import TechnicalIndicators
from core.ml_predictor import LightGBMPredictor
from core.risk_manager import RiskManager

class HighFrequencyStrategy:
    def __init__(self, risk_manager: RiskManager, ml_predictor: LightGBMPredictor):
        self.risk_manager = risk_manager
        self.ml_predictor = ml_predictor
        self.min_confidence = 0.7
        self.signal_buffer = []
        
    def calculate_signal_quality(self, ml_signal: Dict, market_data: Dict, 
                               technical_indicators: Dict) -> float:
        """计算信号质量分数"""
        scores = {
            'ml_confidence': ml_signal['confidence'] * 0.3,
            'technical_alignment': self._check_technical_alignment(technical_indicators) * 0.2,
            'market_microstructure': self._analyze_microstructure(market_data) * 0.2,
            'volatility_score': self._volatility_filter(technical_indicators) * 0.15,
            'volume_profile': self._analyze_volume(market_data) * 0.15
        }
        
        return sum(scores.values())
    
    def _check_technical_alignment(self, indicators: Dict) -> float:
        """检查技术指标对齐度"""
        score = 0
        
        # RSI信号
        if indicators['rsi'] < 30:  # 超卖
            score += 0.33
        elif indicators['rsi'] > 70:  # 超买
            score -= 0.33
            
        # MACD信号
        if indicators['macd_hist'] > 0:
            score += 0.33
        else:
            score -= 0.33
            
        # 布林带位置
        if indicators['bb_position'] < 0.2:  # 接近下轨
            score += 0.34
        elif indicators['bb_position'] > 0.8:  # 接近上轨
            score -= 0.34
            
        return max(0, min(1, (score + 1) / 2))
    
    def _analyze_microstructure(self, market_data: Dict) -> float:
        """分析市场微结构"""
        spread = market_data.get('spread', 0)
        imbalance = market_data.get('imbalance', 0)
        
        # 计算流动性分数
        liquidity_score = 1 / (1 + spread * 1000)  # spread越小越好
        
        # 计算订单流分数
        flow_score = abs(imbalance)  # 不平衡度越大信号越强
        
        return (liquidity_score + flow_score) / 2
    
    def _volatility_filter(self, indicators: Dict) -> float:
        """波动率过滤器"""
        volatility = indicators.get('volatility', 0.02)
        
        # 波动率在适中范围内得分最高
        if 0.01 < volatility < 0.03:
            return 1.0
        elif volatility < 0.005 or volatility > 0.05:
            return 0.2
        else:
            return 0.6
    
    def _analyze_volume(self, market_data: Dict) -> float:
        """分析成交量"""
        volume_ratio = market_data.get('volume_ratio', 1)
        buy_pressure = market_data.get('buy_pressure', 0.5)
        
        # 成交量放大且买压强则得分高
        volume_score = min(volume_ratio / 2, 1)
        pressure_score = abs(buy_pressure - 0.5) * 2
        
        return (volume_score + pressure_score) / 2
    
    def detect_toxic_flow(self, trades_df: pd.DataFrame, 
                         order_book: Dict) -> bool:
        """检测毒性订单流"""
        if trades_df.empty:
            return False
            
        # 计算Kyle's Lambda (价格冲击)
        price_changes = trades_df['price'].diff().abs()
        volumes = trades_df['amount']
        
        kyle_lambda = price_changes.sum() / np.sqrt(volumes.sum())
        
        # 检测是否存在毒性流
        is_toxic = kyle_lambda > 0.001  # 阈值需要根据实际调整
        
        return not is_toxic
    
    def generate_signal(self, market_data: pd.DataFrame, 
                       order_book: Dict, recent_trades: pd.DataFrame) -> Optional[Dict]:
        """生成交易信号"""
        # 计算技术指标
        indicators = TechnicalIndicators.calculate_features(market_data)
        latest_indicators = indicators.iloc[-1].to_dict()
        
        # ML预测
        ml_signal = self.ml_predictor.predict(market_data)
        
        # 计算信号质量
        signal_quality = self.calculate_signal_quality(
            ml_signal, 
            order_book,
            latest_indicators
        )
        
        # 检查是否通过质量阈值
        if signal_quality < self.min_confidence:
            return None
            
        # 检测毒性流
        if not self.detect_toxic_flow(recent_trades, order_book):
            return None
            
        # 计算仓位大小
        recent_returns = market_data['returns'].tail(100)
        position_size = self.risk_manager.calculate_position_size(
            ml_signal, 
            self.risk_manager.calculate_equity(),
            recent_returns
        )
        
        # 风险检查
        current_price = market_data['close'].iloc[-1]
        if not self.risk_manager.check_risk_limits(position_size, current_price):
            return None
            
        return {
            'symbol': 'BTCUSDT',  # 这里应该是动态的
            'side': ml_signal['direction'],
            'size': position_size,
            'price': current_price,
            'confidence': signal_quality,
            'ml_prediction': ml_signal['predicted_return'],
            'timestamp': pd.Timestamp.now()
        }
