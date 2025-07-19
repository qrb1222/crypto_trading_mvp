import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import jit, prange
import pickle
from typing import Dict, List, Tuple

class FastLightGBMPredictor:
    """优化的LightGBM预测器，实现微秒级推理"""
    
    def __init__(self):
        self.models = {}
        self.feature_cache = {}
        self.feature_names = None
        
    @jit(nopython=True)
    def _compute_technical_features(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """使用Numba加速技术指标计算"""
        features = np.empty(20)
        
        # 收益率
        returns = np.diff(prices) / prices[:-1]
        features[0] = returns[-1]
        features[1] = np.mean(returns[-5:])
        features[2] = np.std(returns[-20:])
        
        # RSI快速计算
        gains = np.maximum(returns, 0)
        losses = -np.minimum(returns, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / (avg_loss + 1e-10)
        features[3] = 100 - 100 / (1 + rs)
        
        # 成交量特征
        features[4] = volumes[-1] / np.mean(volumes[-20:])
        features[5] = np.std(volumes[-20:]) / (np.mean(volumes[-20:]) + 1e-10)
        
        # 价格位置
        high_20 = np.max(prices[-20:])
        low_20 = np.min(prices[-20:])
        features[6] = (prices[-1] - low_20) / (high_20 - low_20 + 1e-10)
        
        # 趋势强度
        ma_5 = np.mean(prices[-5:])
        ma_20 = np.mean(prices[-20:])
        features[7] = (ma_5 - ma_20) / ma_20
        
        # 填充剩余特征
        for i in range(8, 20):
            features[i] = 0.0
            
        return features
    
    def prepare_features_fast(self, market_data: Dict) -> np.ndarray:
        """超快速特征准备"""
        prices = market_data['prices']
        volumes = market_data['volumes']
        
        # 使用JIT编译的函数
        technical_features = self._compute_technical_features(prices, volumes)
        
        # 市场微结构特征
        microstructure_features = np.array([
            market_data.get('spread', 0),
            market_data.get('imbalance', 0),
            market_data.get('buy_pressure', 0.5),
            market_data.get('vpin', 0),
            market_data.get('kyle_lambda', 0)
        ])
        
        # 时间特征
        now = pd.Timestamp.now()
        time_features = np.array([
            now.hour / 24,
            now.minute / 60,
            now.dayofweek / 7
        ])
        
        return np.concatenate([technical_features, microstructure_features, time_features])
    
    def predict_ultra_fast(self, features: np.ndarray) -> Dict:
        """超快速预测（目标<10微秒）"""
        # 使用最快的模型进行预测
        model = self.models.get('fast_model')
        if model is None:
            return {'direction': 'HOLD', 'confidence': 0, 'predicted_return': 0}
        
        # 直接使用底层predict
        prediction = model.predict(features.reshape(1, -1), num_threads=1)[0]
        
        # 快速决策
        direction = 'BUY' if prediction > 0.0001 else 'SELL' if prediction < -0.0001 else 'HOLD'
        confidence = min(abs(prediction) * 100, 1.0)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_return': prediction
        }
    
    def train_fast_model(self, X: np.ndarray, y: np.ndarray):
        """训练优化的模型"""
        # 使用更激进的参数以获得更快的推理
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,  # 减少以加快推理
            'max_depth': 5,    # 限制深度
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'min_data_in_leaf': 100,  # 防止过拟合
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'num_threads': 4
        }
        
        train_data = lgb.Dataset(X, label=y)
        
        # 训练
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,  # 限制树的数量
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        self.models['fast_model'] = model
        
        # 转换为更快的格式
        model.save_model('fast_model.txt', num_iteration=model.best_iteration)
