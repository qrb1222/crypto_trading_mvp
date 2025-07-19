import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API配置
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    
    # 交易配置
    SYMBOLS = ['BTCUSDT', 'ETHUSDT']
    BASE_CURRENCY = 'USDT'
    
    # 风险管理
    MAX_LEVERAGE = 30
    MAX_DRAWDOWN = 0.4
    POSITION_SIZE_LIMIT = 0.1
    MIN_CONFIDENCE = 0.7
    KELLY_FRACTION = 0.25
    
    # 模型参数
    LOOKBACK_WINDOW = 100
    PREDICTION_HORIZON = 5
    
    # 路径配置
    DATA_DIR = 'data'
    MODEL_DIR = 'data/models'
    DB_PATH = 'data/market_data.db'
