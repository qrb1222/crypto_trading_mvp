import pandas as pd
import numpy as np
from typing import Tuple

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                                 std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 14) -> pd.Series:
        """计算ATR (Average True Range)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标特征"""
        # 价格特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 技术指标
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
        macd, signal, hist = TechnicalIndicators.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        df['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # 成交量指标
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # 波动率
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['atr'] = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'])
        
        return df
