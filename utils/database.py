import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_tables()
    
    def init_tables(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 市场数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            indicators TEXT,
            UNIQUE(symbol, timestamp)
        )
        ''')
        
        # 交易记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            size REAL NOT NULL,
            confidence REAL,
            ml_prediction REAL,
            timestamp DATETIME NOT NULL,
            pnl REAL,
            status TEXT DEFAULT 'PENDING'
        )
        ''')
        
        # 性能指标表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            equity REAL,
            daily_return REAL,
            trades_count INTEGER,
            win_rate REAL,
            UNIQUE(date)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_market_data(self, symbol: str, data: Dict):
        """保存市场数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if 'ohlcv' in data and not data['ohlcv'].empty:
            latest = data['ohlcv'].iloc[-1]
            
            cursor.execute('''
            INSERT OR REPLACE INTO market_data 
            (symbol, timestamp, open, high, low, close, volume, indicators)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now(),
                latest['open'],
                latest['high'],
                latest['low'],
                latest['close'],
                latest['volume'],
                json.dumps(data.get('indicators', {}))
            ))
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade: Dict):
        """保存交易记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO trades 
        (symbol, side, price, size, confidence, ml_prediction, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['symbol'],
            trade['side'],
            trade['price'],
            trade['size'],
            trade['confidence'],
            trade['ml_prediction'],
            trade['timestamp']
        ))
        
        conn.commit()
        conn.close()
    
    def get_trading_statistics(self) -> Dict:
        """获取交易统计"""
        conn = sqlite3.connect(self.db_path)
        
        # 获取所有交易
        trades_df = pd.read_sql_query(
            "SELECT * FROM trades WHERE status = 'FILLED'", 
            conn
        )
        
        conn.close()
        
        if trades_df.empty:
            return {}
        
        # 计算统计指标
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_return = trades_df['pnl'].mean() if 'pnl' in trades_df else 0
        
        # 计算夏普比率（简化版）
        returns = trades_df['pnl'].values
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': 0  # 需要更复杂的计算
        }
