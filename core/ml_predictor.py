import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class LightGBMPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征"""
        features = df.copy()
        
        # 添加滞后特征
        for lag in [1, 5, 10, 20]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
            
        # 添加滚动统计
        for window in [5, 10, 20]:
            features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
            features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
            features[f'volume_mean_{window}'] = features['volume'].rolling(window).mean()
            
        # 时间特征
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        
        # 删除NaN值
        features = features.dropna()
        
        # 选择特征列
        self.feature_columns = [col for col in features.columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        return features[self.feature_columns]
    
    def train(self, df: pd.DataFrame, target_col: str = 'returns', 
              horizon: int = 5):
        """训练模型"""
        # 准备特征
        features = self.prepare_features(df)
        
        # 创建目标变量（未来收益）
        target = df['returns'].shift(-horizon)
        target = target.loc[features.index]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, shuffle=False
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练LightGBM
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 4
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # 保存模型
        self.models[f'horizon_{horizon}'] = model
        self.scalers[f'horizon_{horizon}'] = scaler
        
        # 特征重要性
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        print(f"Top 10 features:\n{importance.head(10)}")
        
        return model
    
    def predict(self, df: pd.DataFrame, horizon: int = 5) -> Dict:
        """生成预测"""
        model_key = f'horizon_{horizon}'
        if model_key not in self.models:
            raise ValueError(f"Model for horizon {horizon} not trained")
            
        features = self.prepare_features(df)
        latest_features = features.iloc[-1:].values
        
        # 标准化
        scaled_features = self.scalers[model_key].transform(latest_features)
        
        # 预测
        prediction = self.models[model_key].predict(scaled_features)[0]
        
        # 计算置信度（基于预测的绝对值）
        confidence = min(abs(prediction) * 10, 1.0)
        
        return {
            'predicted_return': prediction,
            'direction': 'BUY' if prediction > 0 else 'SELL',
            'confidence': confidence,
            'horizon': horizon
        }
