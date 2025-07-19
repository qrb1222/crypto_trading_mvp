import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Dict

class DataCollector:
    def __init__(self, exchange='binance'):
        self.exchange = getattr(ccxt, exchange)({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                    limit: int = 1000) -> pd.DataFrame:
        """获取K线数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 
                                             'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            return pd.DataFrame()
    
    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """获取订单簿数据"""
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            # 计算订单簿不平衡
            bid_volume = sum([bid[1] for bid in order_book['bids']])
            ask_volume = sum([ask[1] for ask in order_book['asks']])
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # 计算加权中间价
            if order_book['bids'] and order_book['asks']:
                best_bid = order_book['bids'][0][0]
                best_ask = order_book['asks'][0][0]
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2
            else:
                spread = 0
                mid_price = 0
                
            return {
                'mid_price': mid_price,
                'spread': spread,
                'imbalance': imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return {}
    
    def fetch_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """获取最近成交记录"""
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 计算买卖压力
            buy_volume = df[df['side'] == 'buy']['amount'].sum()
            sell_volume = df[df['side'] == 'sell']['amount'].sum()
            
            return df, {'buy_pressure': buy_volume / (buy_volume + sell_volume)}
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return pd.DataFrame(), {}
