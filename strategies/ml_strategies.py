import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MLStrategy(BaseStrategy):
    """Base class for machine learning strategies"""
    
    def __init__(self, name: str, parameters: Dict = None):
        super().__init__(name, parameters)
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning"""
        df = data.copy()
        
        # Technical indicators
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['ma_5'] = self.calculate_sma(df['close'], 5)
        df['ma_20'] = self.calculate_sma(df['close'], 20)
        df['ma_50'] = self.calculate_sma(df['close'], 50)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['returns_lag1'] = df['returns'].shift(1)
        df['returns_lag2'] = df['returns'].shift(2)
        df['returns_lag3'] = df['returns'].shift(3)
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # High-low ratio
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Support and resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        return df
    
    def create_target(self, data: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Create target variable for prediction"""
        future_returns = data['close'].shift(-horizon) / data['close'] - 1
        
        # Create categorical target: 0=hold, 1=buy, 2=sell
        target = pd.Series(0, index=data.index)  # Default: hold
        
        threshold = self.get_parameter('prediction_threshold', 0.02)
        target[future_returns > threshold] = 1  # Buy
        target[future_returns < -threshold] = 2  # Sell
        
        return target
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for training"""
        feature_cols = [
            'rsi', 'ma_5', 'ma_20', 'ma_50', 'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'returns_lag1', 'returns_lag2', 'returns_lag3',
            'volume_ratio', 'volatility', 'momentum_5', 'momentum_10', 'momentum_20',
            'hl_ratio', 'price_position'
        ]
        
        # Filter out columns that don't exist or have too many NaNs
        available_features = []
        for col in feature_cols:
            if col in df.columns and df[col].notna().sum() > len(df) * 0.8:
                available_features.append(col)
        
        return available_features
    
    def train_model(self, data: pd.DataFrame) -> Dict:
        """Train the machine learning model"""
        try:
            df = self.prepare_features(data)
            self.feature_columns = self.select_features(df)
            
            if len(self.feature_columns) == 0:
                return {'success': False, 'error': 'No valid features available'}
            
            # Create target
            target = self.create_target(df)
            
            # Remove rows with NaN values
            mask = df[self.feature_columns].notna().all(axis=1) & target.notna()
            X = df.loc[mask, self.feature_columns]
            y = target.loc[mask]
            
            if len(X) < 100:
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Evaluate model using time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                scores.append(accuracy_score(y_test, y_pred))
            
            avg_score = np.mean(scores)
            
            return {
                'success': True, 
                'accuracy': avg_score,
                'features_used': len(self.feature_columns),
                'training_samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        """Make prediction using trained model"""
        try:
            if not self.is_trained:
                return 0, 0.0
            
            df = self.prepare_features(data)
            
            # Use only the last row for prediction
            last_row = df.iloc[-1:][self.feature_columns]
            
            if last_row.isna().any().any():
                return 0, 0.0
            
            X_scaled = self.scaler.transform(last_row)
            prediction = self.model.predict(X_scaled)[0]
            
            # Get prediction probability if available
            try:
                probabilities = self.model.predict_proba(X_scaled)[0]
                confidence = np.max(probabilities)
            except:
                confidence = 0.6  # Default confidence
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0, 0.0


class RandomForestStrategy(MLStrategy):
    """Random Forest based trading strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'prediction_threshold': 0.02,
            'confidence_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("RandomForest", default_params)
        
        self.model = RandomForestClassifier(
            n_estimators=self.get_parameter('n_estimators', 100),
            max_depth=self.get_parameter('max_depth', 10),
            min_samples_split=self.get_parameter('min_samples_split', 5),
            random_state=42
        )
    
    def get_minimum_data_length(self) -> int:
        return 100
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators (handled in prepare_features)"""
        return self.prepare_features(data)
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate RF-based trading signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            # Train model if not trained
            if not self.is_trained:
                train_result = self.train_model(data)
                if not train_result['success']:
                    return {'action': 'hold', 'confidence': 0, 'reasoning': f"Training failed: {train_result['error']}"}
            
            # Make prediction
            prediction, confidence = self.predict(data)
            confidence_threshold = self.get_parameter('confidence_threshold', 0.6)
            
            if confidence < confidence_threshold:
                return {'action': 'hold', 'confidence': confidence, 'reasoning': 'Low confidence prediction'}
            
            signal = {'action': 'hold', 'confidence': confidence, 'reasoning': ''}
            
            if prediction == 1:  # Buy signal
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': current_price * 0.95,
                    'take_profit': current_price * 1.10,
                    'reasoning': f'RF buy prediction (confidence: {confidence:.3f})'
                }
            elif prediction == 2:  # Sell signal
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': current_price * 1.05,
                    'take_profit': current_price * 0.90,
                    'reasoning': f'RF sell prediction (confidence: {confidence:.3f})'
                }
            else:  # Hold
                signal = {
                    'action': 'hold',
                    'confidence': confidence,
                    'reasoning': f'RF hold prediction (confidence: {confidence:.3f})'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating RF signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class GradientBoostingStrategy(MLStrategy):
    """Gradient Boosting based trading strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'prediction_threshold': 0.02,
            'confidence_threshold': 0.65
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("GradientBoosting", default_params)
        
        self.model = GradientBoostingClassifier(
            n_estimators=self.get_parameter('n_estimators', 100),
            learning_rate=self.get_parameter('learning_rate', 0.1),
            max_depth=self.get_parameter('max_depth', 6),
            random_state=42
        )
    
    def get_minimum_data_length(self) -> int:
        return 100
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators (handled in prepare_features)"""
        return self.prepare_features(data)
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate GB-based trading signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            # Train model if not trained
            if not self.is_trained:
                train_result = self.train_model(data)
                if not train_result['success']:
                    return {'action': 'hold', 'confidence': 0, 'reasoning': f"Training failed: {train_result['error']}"}
            
            # Make prediction
            prediction, confidence = self.predict(data)
            confidence_threshold = self.get_parameter('confidence_threshold', 0.65)
            
            if confidence < confidence_threshold:
                return {'action': 'hold', 'confidence': confidence, 'reasoning': 'Low confidence prediction'}
            
            signal = {'action': 'hold', 'confidence': confidence, 'reasoning': ''}
            
            if prediction == 1:  # Buy signal
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': current_price * 0.95,
                    'take_profit': current_price * 1.10,
                    'reasoning': f'GB buy prediction (confidence: {confidence:.3f})'
                }
            elif prediction == 2:  # Sell signal
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': current_price * 1.05,
                    'take_profit': current_price * 0.90,
                    'reasoning': f'GB sell prediction (confidence: {confidence:.3f})'
                }
            else:  # Hold
                signal = {
                    'action': 'hold',
                    'confidence': confidence,
                    'reasoning': f'GB hold prediction (confidence: {confidence:.3f})'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating GB signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class LogisticRegressionStrategy(MLStrategy):
    """Logistic Regression based trading strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'prediction_threshold': 0.02,
            'confidence_threshold': 0.6
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("LogisticRegression", default_params)
        
        self.model = LogisticRegression(
            C=self.get_parameter('C', 1.0),
            max_iter=self.get_parameter('max_iter', 1000),
            random_state=42
        )
    
    def get_minimum_data_length(self) -> int:
        return 100
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators (handled in prepare_features)"""
        return self.prepare_features(data)
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate LR-based trading signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            # Train model if not trained
            if not self.is_trained:
                train_result = self.train_model(data)
                if not train_result['success']:
                    return {'action': 'hold', 'confidence': 0, 'reasoning': f"Training failed: {train_result['error']}"}
            
            # Make prediction
            prediction, confidence = self.predict(data)
            confidence_threshold = self.get_parameter('confidence_threshold', 0.6)
            
            if confidence < confidence_threshold:
                return {'action': 'hold', 'confidence': confidence, 'reasoning': 'Low confidence prediction'}
            
            signal = {'action': 'hold', 'confidence': confidence, 'reasoning': ''}
            
            if prediction == 1:  # Buy signal
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': current_price * 0.95,
                    'take_profit': current_price * 1.08,
                    'reasoning': f'LR buy prediction (confidence: {confidence:.3f})'
                }
            elif prediction == 2:  # Sell signal
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': current_price * 1.05,
                    'take_profit': current_price * 0.92,
                    'reasoning': f'LR sell prediction (confidence: {confidence:.3f})'
                }
            else:  # Hold
                signal = {
                    'action': 'hold',
                    'confidence': confidence,
                    'reasoning': f'LR hold prediction (confidence: {confidence:.3f})'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating LR signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class EnsembleStrategy(BaseStrategy):
    """Ensemble of multiple ML strategies"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'strategies': ['RandomForest', 'GradientBoosting', 'LogisticRegression'],
            'weights': [0.4, 0.4, 0.2],
            'min_agreement': 0.6,
            'confidence_threshold': 0.65
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("MLEnsemble", default_params)
        
        # Initialize sub-strategies
        self.strategies = {
            'RandomForest': RandomForestStrategy(),
            'GradientBoosting': GradientBoostingStrategy(),
            'LogisticRegression': LogisticRegressionStrategy()
        }
    
    def get_minimum_data_length(self) -> int:
        return 100
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for ensemble"""
        return data
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate ensemble signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            strategy_names = self.get_parameter('strategies', ['RandomForest', 'GradientBoosting', 'LogisticRegression'])
            weights = self.get_parameter('weights', [0.4, 0.4, 0.2])
            min_agreement = self.get_parameter('min_agreement', 0.6)
            confidence_threshold = self.get_parameter('confidence_threshold', 0.65)
            
            signals = []
            valid_strategies = 0
            
            # Get signals from all strategies
            for i, strategy_name in enumerate(strategy_names):
                if strategy_name in self.strategies:
                    strategy = self.strategies[strategy_name]
                    signal = strategy.generate_signal(data, current_price, portfolio)
                    
                    if signal['confidence'] > 0:
                        signals.append({
                            'action': signal['action'],
                            'confidence': signal['confidence'],
                            'weight': weights[i] if i < len(weights) else 1.0/len(strategy_names)
                        })
                        valid_strategies += 1
            
            if valid_strategies == 0:
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'No valid strategy signals'}
            
            # Calculate weighted consensus
            buy_score = sum(s['confidence'] * s['weight'] for s in signals if s['action'] == 'buy')
            sell_score = sum(s['confidence'] * s['weight'] for s in signals if s['action'] == 'sell')
            hold_score = sum(s['confidence'] * s['weight'] for s in signals if s['action'] == 'hold')
            
            # Normalize scores
            total_score = buy_score + sell_score + hold_score
            if total_score > 0:
                buy_score /= total_score
                sell_score /= total_score
                hold_score /= total_score
            
            # Determine final action
            max_score = max(buy_score, sell_score, hold_score)
            
            if max_score < confidence_threshold:
                return {'action': 'hold', 'confidence': max_score, 'reasoning': 'Low ensemble confidence'}
            
            if buy_score == max_score and buy_score >= min_agreement:
                signal = {
                    'action': 'buy',
                    'confidence': buy_score,
                    'stop_loss': current_price * 0.95,
                    'take_profit': current_price * 1.10,
                    'reasoning': f'Ensemble buy signal (score: {buy_score:.3f}, strategies: {valid_strategies})'
                }
            elif sell_score == max_score and sell_score >= min_agreement:
                signal = {
                    'action': 'sell',
                    'confidence': sell_score,
                    'stop_loss': current_price * 1.05,
                    'take_profit': current_price * 0.90,
                    'reasoning': f'Ensemble sell signal (score: {sell_score:.3f}, strategies: {valid_strategies})'
                }
            else:
                signal = {
                    'action': 'hold',
                    'confidence': max_score,
                    'reasoning': f'Ensemble hold signal (max score: {max_score:.3f}, agreement: {max_score:.3f})'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating ensemble signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}