import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List

from config import Config
from core.data_collector import DataCollector
from core.fast_predictor import FastLightGBMPredictor
from core.ppo_agent import PPOAgent
from core.enhanced_risk_manager import EnhancedRiskManager
from strategies.enhanced_hf_strategy import EnhancedHighFrequencyStrategy
from utils.logger import setup_logger
from utils.performance_monitor import PerformanceMonitor

class OptimizedTradingSystem:
    """生产级高频交易系统"""
    
    def __init__(self, mode: str = 'paper'):
        self.mode = mode
        self.logger = setup_logger('optimized_trading')
        
        # 核心组件
        self.data_collector = DataCollector()
        self.strategy = EnhancedHighFrequencyStrategy()
        self.risk_manager = EnhancedRiskManager()
        self.performance_monitor = PerformanceMonitor()
        
        # PPO相关
        self.ppo_agent = PPOAgent(state_dim=30)
        self.ppo_update_frequency = 100  # 每100步更新
        self.step_count = 0
        
        # 性能优化
        self.use_cache = True
        self.cache_ttl = 0.1  # 100ms缓存
        self.last_cache_time = {}
        self.cache_data = {}
        
    async def collect_market_data_fast(self, symbol: str) -> Dict:
        """快速数据收集，使用缓存"""
        cache_key = f"{symbol}_market_data"
        now = time.time()
        
        # 检查缓存
        if self.use_cache and cache_key in self.cache_data:
            if now - self.last_cache_time.get(cache_key, 0) < self.cache_ttl:
                return self.cache_data[cache_key]
        
        # 并行收集数据
        tasks = [
            self.data_collector.fetch_ohlcv(symbol, '1m', limit=100),
            self.data_collector.fetch_order_book(symbol),
            self.data_collector.fetch_recent_trades(symbol)
        ]
        
        ohlcv, order_book, (trades, trade_stats) = await asyncio.gather(*[
            asyncio.create_task(asyncio.to_thread(task))
            for task in tasks
        ])
        
        # 计算衍生指标
        if not ohlcv.empty:
            returns = ohlcv['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            market_data = {
                'symbol': symbol,
                'prices': ohlcv['close'].values,
                'volumes': ohlcv['volume'].values,
                'returns_60min': returns.tail(60).values,
                'last_return': returns.iloc[-1],
                'mean_return_60min': returns.tail(60).mean(),
                'std_return_60min': returns.tail(60).std(),
                'volatility': volatility,
                'spread': order_book.get('spread', 0),
                'avg_spread': 0.0001,  # 应该从历史计算
                'imbalance': order_book.get('imbalance', 0),
                'volume': ohlcv['volume'].iloc[-1],
                'avg_volume': ohlcv['volume'].tail(20).mean(),
                'buy_pressure': trade_stats.get('buy_pressure', 0.5),
                'current_position': 0,  # 从仓位管理获取
                'unrealized_pnl': 0,
                'total_equity': 100000
            }
            
            # 更新缓存
            self.cache_data[cache_key] = market_data
            self.last_cache_time[cache_key] = now
            
            return market_data
        
        return {}
    
    async def execute_trade_fast(self, signal: Dict) -> Dict:
        """快速执行交易"""
        start_time = time.perf_counter()
        
        # 风险检查
        position_size = self.risk_manager.calculate_optimal_position_size(
            signal, 
            {'volatility': 0.02, 'regime': 'trending'}  # 简化
        )
        
        if not self.risk_manager.should_take_trade(signal, {}):
            self.logger.info(f"Trade rejected by risk manager: {signal['quality_score']:.3f}")
            return {'status': 'rejected'}
        
        # 计算实际交易参数
        trade_params = {
            'symbol': signal['symbol'],
            'side': signal['direction'],
            'size': position_size * self.risk_manager.max_leverage,
            'price': signal.get('price', 0),
            'leverage': self.risk_manager.max_leverage,
            'confidence': signal['quality_score'],
            'predicted_return': signal['predicted_return'],
            'execution_time': time.perf_counter() - start_time
        }
        
        if self.mode == 'paper':
            # 模拟执行
            self.logger.info(
                f"PAPER: {trade_params['side']} {trade_params['size']:.4f} "
                f"@ leverage {trade_params['leverage']}x "
                f"(conf: {trade_params['confidence']:.3f}, "
                f"exe: {trade_params['execution_time']*1000:.1f}ms)"
            )
            
            # 记录交易
            trade_result = {
                **trade_params,
                'status': 'filled',
                'timestamp': pd.Timestamp.now(),
                'pnl': 0  # 稍后计算
            }
            
            # 更新PPO经验
            self._update_ppo_experience(signal, trade_result)
            
            return trade_result
        else:
            # 实盘执行
            try:
                order = await self._place_real_order(trade_params)
                return order
            except Exception as e:
                self.logger.error(f"Order execution failed: {e}")
                return {'status': 'failed', 'error': str(e)}
    
    def _update_ppo_experience(self, signal: Dict, trade_result: Dict):
        """更新PPO经验"""
        # 计算奖励（简化版）
        if trade_result['status'] == 'filled':
            reward = trade_result.get('pnl', 0) * 100  # 放大奖励
        else:
            reward = -0.1  # 小惩罚
        
        # 存储经验
        state = signal.get('state', np.zeros(30))
        action = signal['ppo_action']
        value = signal['ppo_value']
        log_prob = 0  # 需要从PPO获取
        
        self.ppo_agent.store_transition(state, action, reward, value, log_prob, False)
        
        # 定期更新
        self.step_count += 1
        if self.step_count % self.ppo_update_frequency == 0:
            next_value = 0  # 简化
            self.ppo_agent.update(next_value)
    
    async def trading_loop_optimized(self):
        """优化的主交易循环"""
        self.logger.info(f"Starting optimized trading loop in {self.mode} mode")
        
        # 预热
        await self._warmup()
        
        loop_count = 0
        total_execution_time = 0
        
        while True:
            loop_start = time.perf_counter()
            
            try:
                # 并行处理多个交易对
                tasks = []
                for symbol in Config.SYMBOLS[:2]:  # 限制并发
                    task = self._process_symbol(symbol)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 性能监控
                loop_time = time.perf_counter() - loop_start
                total_execution_time += loop_time
                loop_count += 1
                
                if loop_count % 100 == 0:
                    avg_time = total_execution_time / loop_count * 1000
                    self.logger.info(f"Avg loop time: {avg_time:.2f}ms")
                    
                    # 打印性能统计
                    self.performance_monitor.print_stats()
                
                # 动态休眠
                sleep_time = max(0, 0.1 - loop_time)  # 目标100ms循环
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                self.logger.info("Graceful shutdown initiated")
                break
            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_symbol(self, symbol: str):
        """处理单个交易对"""
        try:
            # 收集数据
            market_data = await self.collect_market_data_fast(symbol)
            
            if not market_data:
                return None
            
            # 生成信号
            signal = await self.strategy.generate_ultra_fast_signal(market_data)
            
            if signal and signal['quality_score'] >= 0.75:
                # 执行交易
                result = await self.execute_trade_fast(signal)
                
                # 更新性能
                if result.get('status') == 'filled':
                    self.performance_monitor.record_trade(result)
                    self.risk_manager.update_performance(result)
                    
                return result
                
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return None
    
    async def _warmup(self):
        """系统预热"""
        self.logger.info("System warming up...")
        
        # 加载模型
        try:
            self.strategy.ml_predictor.models['fast_model'] = lgb.Booster(
                model_file='fast_model.txt'
            )
            self.logger.info("Model loaded successfully")
        except:
            self.logger.warning("No pre-trained model found, using defaults")
        
        # 预热数据收集
        for symbol in Config.SYMBOLS:
            await self.collect_market_data_fast(symbol)
        
        self.logger.info("Warmup completed")

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.trades = []
        self.start_equity = 100000
        self.current_equity = 100000
        
    def record_trade(self, trade: Dict):
        """记录交易"""
        self.trades.append(trade)
        
        # 简化的盈亏计算
        if trade.get('pnl'):
            self.current_equity += trade['pnl']
    
    def print_stats(self):
        """打印统计信息"""
        if not self.trades:
            return
        
        recent_trades = self.trades[-100:]
        wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        total = len(recent_trades)
        
        win_rate = wins / total if total > 0 else 0
        total_return = (self.current_equity - self.start_equity) / self.start_equity
        
        print(f"\n=== Performance Stats ===")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Current Equity: ${self.current_equity:,.2f}")
        print("=======================\n")

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    
    system = OptimizedTradingSystem(mode=args.mode)
    
    if args.train:
        # 训练流程
        print("Training models...")
        # TODO: 实现训练
    else:
        # 交易流程
        try:
            await system.trading_loop_optimized()
        except KeyboardInterrupt:
            print("\nShutdown complete")
        finally:
            system.performance_monitor.print_stats()

if __name__ == "__main__":
    # 设置异步事件循环策略
    import platform
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
