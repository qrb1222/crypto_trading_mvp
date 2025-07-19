import numpy as np
import pandas as pd
from arch import arch_model
from typing import Dict, Tuple

class RiskManager:
    def __init__(self, max_leverage: float = 30, max_drawdown: float = 0.4):
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.positions = {}
        self.equity_curve = []
        
    def calculate_position_size(self, signal: Dict, capital: float, 
                              recent_returns: pd.Series) -> float:
        """使用动态Kelly准则计算仓位大小"""
        # 计算历史胜率和盈亏比
        wins = recent_returns[recent_returns > 0]
        losses = recent_returns[recent_returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0
            
        win_rate = len(wins) / len(recent_returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        # Kelly公式
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # 分数Kelly（使用1/4降低风险）
        kelly_fraction = 0.25
        
        # 波动率调整
        volatility = recent_returns.std()
        vol_adjustment = 1 / (1 + volatility * 10)
        
        # 置信度调整
        confidence = signal.get('confidence', 0.5)
        
        # 最终仓位
        position_size = kelly * kelly_fraction * confidence * vol_adjustment
        
        # 限制最大仓位
        position_size = min(position_size, 0.1)  # 最大10%
        position_size = max(position_size, 0)    # 不能为负
        
        return position_size * capital
    
    def calculate_garch_volatility(self, returns: pd.Series) -> float:
        """使用GARCH模型估计波动率"""
        try:
            # 转换为百分比收益率
            returns_pct = returns * 100
            
            # 拟合GARCH(1,1)模型
            model = arch_model(returns_pct, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            # 预测下一期波动率
            forecast = model_fit.forecast(horizon=1)
            volatility = np.sqrt(forecast.variance.values[-1, :][0]) / 100
            
            return volatility
        except Exception as e:
            print(f"GARCH fitting error: {e}")
            # 使用简单历史波动率作为后备
            return returns.std()
    
    def calculate_var_cvar(self, returns: pd.Series, 
                          confidence_level: float = 0.95) -> Tuple[float, float]:
        """计算VaR和CVaR"""
        # 计算VaR
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # 计算CVaR（条件VaR）
        cvar = returns[returns <= var].mean()
        
        return var, cvar
    
    def check_risk_limits(self, position_size: float, current_price: float) -> bool:
        """检查风险限制"""
        # 计算当前总仓位
        total_position = sum([pos['size'] * pos['current_price'] 
                            for pos in self.positions.values()])
        
        # 检查杠杆
        leverage = (total_position + position_size) / self.calculate_equity()
        if leverage > self.max_leverage:
            print(f"Leverage limit exceeded: {leverage:.2f}")
            return False
            
        # 检查回撤
        if len(self.equity_curve) > 0:
            peak = max(self.equity_curve)
            current_equity = self.calculate_equity()
            drawdown = (peak - current_equity) / peak
            
            if drawdown > self.max_drawdown:
                print(f"Drawdown limit exceeded: {drawdown:.2%}")
                return False
                
        return True
    
    def update_position(self, symbol: str, size: float, 
                       entry_price: float, side: str):
        """更新仓位"""
        self.positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'current_price': entry_price,
            'side': side,
            'timestamp': pd.Timestamp.now()
        }
    
    def calculate_equity(self) -> float:
        """计算当前权益"""
        # 这里简化处理，实际应该从交易所获取
        base_equity = 100000  # 初始资金
        
        # 计算未实现盈亏
        unrealized_pnl = 0
        for symbol, pos in self.positions.items():
            if pos['side'] == 'BUY':
                unrealized_pnl += pos['size'] * (pos['current_price'] - pos['entry_price'])
            else:
                unrealized_pnl += pos['size'] * (pos['entry_price'] - pos['current_price'])
                
        return base_equity + unrealized_pnl
