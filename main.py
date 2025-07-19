import asyncio
import argparse
import sys
from datetime import datetime
import pandas as pd
from typing import Dict

from config import Config
from core.data_collector import DataCollector
from core.ml_predictor import LightGBMPredictor
from core.risk_manager import RiskManager
from strategies.high_frequency import HighFrequencyStrategy
from utils.logger import setup_logger
from utils.database import Database

class TradingSystem:
    def __init__(self, mode: str = 'paper'):
        self.mode = mode
        self.logger = setup_logger('trading_system')
        self.db = Database(Config.DB_PATH)
        
        # 初始化组件
        self.data_collector = DataCollector()
        self.ml_predictor = LightGBMPredictor()
        self.risk_manager = RiskManager(
            max_leverage=Config.MAX_LEVERAGE,
            max_drawdown=Config.MAX_DRAWDOWN
        )
        self.strategy = HighFrequencyStrategy(self.risk_manager, self.ml_predictor)
        
        self.is_running = False
        
    def train_models(self):
        """训练机器学习模型"""
        self.logger.info("Starting model training...")
        
        for symbol in Config.SYMBOLS:
            self.logger.info(f"Training model for {symbol}")
            
            # 获取历史数据
            df = self.data_collector.fetch_ohlcv(symbol, '1m', limit=10000)
            
            if df.empty:
                self.logger.error(f"No data available for {symbol}")
                continue
                
            # 训练模型
            self.ml_predictor.train(df, horizon=5)
            
        self.logger.info("Model training completed")
        
    async def collect_market_data(self, symbol: str) -> Dict:
        """收集实时市场数据"""
        # K线数据
        ohlcv = self.data_collector.fetch_ohlcv(symbol, '1m', limit=200)
        
        # 订单簿数据
        order_book = self.data_collector.fetch_order_book(symbol)
        
        # 最近成交
        trades, trade_stats = self.data_collector.fetch_recent_trades(symbol)
        
        # 更新市场数据
        market_data = {
            'ohlcv': ohlcv,
            'order_book': order_book,
            'trades': trades,
            'trade_stats': trade_stats
        }
        
        # 保存到数据库
        self.db.save_market_data(symbol, market_data)
        
        return market_data
    
    async def execute_trade(self, signal: Dict):
        """执行交易（模拟或实盘）"""
        self.logger.info(f"Executing trade signal: {signal}")
        
        if self.mode == 'paper':
            # 模拟交易
            self.logger.info(f"PAPER TRADE: {signal['side']} {signal['size']:.4f} @ {signal['price']:.2f}")
            
            # 更新仓位
            self.risk_manager.update_position(
                signal['symbol'],
                signal['size'],
                signal['price'],
                signal['side']
            )
            
            # 记录交易
            self.db.save_trade(signal)
            
        else:
            # 实盘交易
            try:
                if signal['side'] == 'BUY':
                    order = self.data_collector.exchange.create_market_buy_order(
                        signal['symbol'],
                        signal['size']
                    )
                else:
                    order = self.data_collector.exchange.create_market_sell_order(
                        signal['symbol'],
                        signal['size']
                    )
                    
                self.logger.info(f"Order executed: {order}")
                
            except Exception as e:
                self.logger.error(f"Order execution failed: {e}")
    
    async def trading_loop(self):
        """主交易循环"""
        self.logger.info(f"Starting trading loop in {self.mode} mode")
        self.is_running = True
        
        while self.is_running:
            try:
                for symbol in Config.SYMBOLS:
                    # 收集市场数据
                    market_data = await self.collect_market_data(symbol)
                    
                    # 生成交易信号
                    signal = self.strategy.generate_signal(
                        market_data['ohlcv'],
                        market_data['order_book'],
                        market_data['trades']
                    )
                    
                    # 执行交易
                    if signal:
                        await self.execute_trade(signal)
                    
                # 短暂休眠避免超限
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                self.is_running = False
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    def print_statistics(self):
        """打印交易统计"""
        stats = self.db.get_trading_statistics()
        
        print("\n=== Trading Statistics ===")
        print(f"Total Trades: {stats.get('total_trades', 0)}")
        print(f"Win Rate: {stats.get('win_rate', 0):.2%}")
        print(f"Average Return: {stats.get('avg_return', 0):.4f}")
        print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {stats.get('max_drawdown', 0):.2%}")
        print("========================\n")

async def main():
    parser = argparse.ArgumentParser(description='Crypto Trading System')
    parser.add_argument('--mode', choices=['paper', 'live'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--train', action='store_true', 
                       help='Train models before trading')
    parser.add_argument('--stats', action='store_true',
                       help='Show trading statistics')
    
    args = parser.parse_args()
    
    # 创建交易系统
    system = TradingSystem(mode=args.mode)
    
    # 训练模型
    if args.train:
        system.train_models()
    
    # 显示统计
    if args.stats:
        system.print_statistics()
        return
    
    # 开始交易
    try:
        await system.trading_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        system.print_statistics()

if __name__ == "__main__":
    asyncio.run(main())
