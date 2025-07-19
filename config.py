import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API配置
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    
    # 交易配置 - 优化版
    SYMBOLS = ['BTCUSDT', 'ETHUSDT']  # 专注主流币种
    BASE_CURRENCY = 'USDT'
    
    # 风险管理 - 更严格
    MAX_LEVERAGE = 30
    MAX_DRAWDOWN = 0.35  # 降低到35%
    POSITION_SIZE_LIMIT = 0.08  # 单仓位最大8%
    MIN_CONFIDENCE = 0.75  # 提高到75%
    KELLY_FRACTION = 0.2  # 更保守
    
    # 性能目标
    TARGET_WIN_RATE = 0.75  # 75%胜率
    TARGET_SHARPE = 2.0
    TARGET_ANNUAL_RETURN = 2.0  # 200%
    
    # 执行参数
    MAX_SLIPPAGE = 0.001  # 0.1%
    EXECUTION_TIMEOUT = 0.5  # 500ms
    
    # 系统参数
    UPDATE_FREQUENCY = 100  # ms
    MAX_CONCURRENT_TRADES = 3
    CACHE_TTL = 0.1  # 100ms
