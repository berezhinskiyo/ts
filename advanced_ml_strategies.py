#!/usr/bin/env python3
"""
Продвинутые ML стратегии с использованием ARIMA, LSTM и других методов
для прогнозирования временных рядов
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ML библиотеки
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Временные ряды
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

# LSTM (если доступен)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("TensorFlow не установлен. LSTM стратегии будут пропущены.")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Результат прогнозирования"""
    predicted_price: float
    confidence: float
    method: str
    features_used: List[str]
    model_score: float

class AdvancedMLStrategies:
    """Продвинутые ML стратегии для торговли"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        
        # Параметры управления рисками
        self.max_risk_per_trade = 0.02
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.15
        
        # ML модели
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Параметры для временных рядов
        self.lookback_window = 20
        self.forecast_horizon = 5
        
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создание технических индикаторов"""
        try:
            df = data.copy()
            
            # Простые скользящие средние
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Экспоненциальные скользящие средние
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_10'] = df['close'].ewm(span=10).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['momentum_20'] = df['close'].pct_change(20)
            
            # Volatility
            df['volatility_5'] = df['close'].rolling(window=5).std()
            df['volatility_20'] = df['close'].rolling(window=20).std()
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
            
            # Price patterns
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания технических индикаторов: {e}")
            return data
    
    def arima_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """ARIMA стратегия для прогнозирования цен"""
        try:
            logger.info(f"Тестирование ARIMA стратегии для {symbol}")
            
            # Подготавливаем данные
            df = self.create_technical_features(data)
            df = df.dropna()
            
            if len(df) < 50:
                return None
            
            current_capital = self.initial_capital
            positions = {}
            trades = []
            equity_history = []
            
            for i in range(50, len(df)):
                current_price = df.iloc[i]['close']
                
                # Проверяем стоп-ордера
                if symbol in positions:
                    self.check_stop_orders(symbol, current_price, positions, trades, current_capital)
                
                # Получаем исторические данные для ARIMA
                price_series = df.iloc[i-50:i]['close'].values
                
                try:
                    # Подбираем параметры ARIMA автоматически
                    best_aic = float('inf')
                    best_model = None
                    best_order = None
                    
                    # Перебираем различные комбинации параметров
                    for p in range(0, 3):
                        for d in range(0, 2):
                            for q in range(0, 3):
                                try:
                                    model = ARIMA(price_series, order=(p, d, q))
                                    fitted_model = model.fit()
                                    if fitted_model.aic < best_aic:
                                        best_aic = fitted_model.aic
                                        best_model = fitted_model
                                        best_order = (p, d, q)
                                except:
                                    continue
                    
                    if best_model is not None:
                        # Делаем прогноз на следующий день
                        forecast = best_model.forecast(steps=1)
                        predicted_price = forecast[0]
                        
                        # Рассчитываем уверенность на основе стандартной ошибки
                        confidence = max(0, min(1, 1 - abs(predicted_price - current_price) / current_price))
                        
                        # Сигнал на покупку/продажу
                        price_change_pct = (predicted_price - current_price) / current_price
                        
                        # Покупка при прогнозе роста > 2%
                        if price_change_pct > 0.02 and symbol not in positions and confidence > 0.6:
                            position_size = self.calculate_position_size(current_price)
                            if position_size > 0:
                                positions[symbol] = {
                                    'size': position_size,
                                    'entry_price': current_price,
                                    'entry_date': df.index[i],
                                    'predicted_price': predicted_price,
                                    'confidence': confidence,
                                    'method': 'ARIMA'
                                }
                                logger.info(f"ARIMA покупка {symbol}: {position_size} акций по {current_price:.2f}₽ (прогноз: {predicted_price:.2f}₽)")
                        
                        # Продажа при прогнозе падения > 1%
                        elif price_change_pct < -0.01 and symbol in positions:
                            position = positions[symbol]
                            exit_price = current_price
                            pnl = (exit_price - position['entry_price']) * position['size']
                            
                            trades.append({
                                'entry_date': position['entry_date'],
                                'exit_date': df.index[i],
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'return_pct': (exit_price / position['entry_price'] - 1) * 100,
                                'method': 'ARIMA',
                                'confidence': confidence
                            })
                            
                            current_capital += pnl
                            del positions[symbol]
                            logger.info(f"ARIMA продажа {symbol}: PnL={pnl:.2f}₽")
                
                except Exception as e:
                    logger.debug(f"Ошибка ARIMA для {symbol}: {e}")
                
                # Обновляем метрики
                current_equity = current_capital
                if symbol in positions:
                    position = positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_history.append({
                    'date': df.index[i],
                    'equity': current_equity
                })
            
            return self._calculate_metrics(symbol, equity_history, trades, "ARIMA_Strategy")
            
        except Exception as e:
            logger.error(f"Ошибка в ARIMA стратегии для {symbol}: {e}")
            return None
    
    def lstm_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """LSTM стратегия для прогнозирования цен"""
        if not LSTM_AVAILABLE:
            logger.warning("LSTM недоступен. Пропускаем LSTM стратегию.")
            return None
        
        try:
            logger.info(f"Тестирование LSTM стратегии для {symbol}")
            
            # Подготавливаем данные
            df = self.create_technical_features(data)
            df = df.dropna()
            
            if len(df) < 100:
                return None
            
            current_capital = self.initial_capital
            positions = {}
            trades = []
            equity_history = []
            
            # Подготавливаем данные для LSTM
            features = ['close', 'volume', 'sma_5', 'sma_20', 'rsi', 'macd', 'bb_position', 'stoch_k']
            feature_data = df[features].values
            
            # Нормализация данных
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(feature_data)
            
            # Создаем последовательности для LSTM
            def create_sequences(data, lookback):
                X, y = [], []
                for i in range(lookback, len(data)):
                    X.append(data[i-lookback:i])
                    y.append(data[i, 0])  # Предсказываем цену закрытия
                return np.array(X), np.array(y)
            
            lookback = 20
            X, y = create_sequences(scaled_data, lookback)
            
            for i in range(lookback + 50, len(df)):
                current_price = df.iloc[i]['close']
                
                # Проверяем стоп-ордера
                if symbol in positions:
                    self.check_stop_orders(symbol, current_price, positions, trades, current_capital)
                
                try:
                    # Обучаем LSTM на исторических данных
                    train_size = min(200, i - lookback - 50)
                    if train_size > 50:
                        X_train = X[i-train_size-lookback:i-lookback]
                        y_train = y[i-train_size-lookback:i-lookback]
                        
                        # Создаем модель LSTM
                        model = Sequential([
                            LSTM(50, return_sequences=True, input_shape=(lookback, len(features))),
                            Dropout(0.2),
                            LSTM(50, return_sequences=False),
                            Dropout(0.2),
                            Dense(25),
                            Dense(1)
                        ])
                        
                        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                        
                        # Обучаем модель
                        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
                        
                        # Делаем прогноз
                        last_sequence = scaled_data[i-lookback:i].reshape(1, lookback, len(features))
                        predicted_scaled = model.predict(last_sequence, verbose=0)[0, 0]
                        
                        # Обратно преобразуем в цену
                        predicted_price = scaler.inverse_transform([[predicted_scaled, 0, 0, 0, 0, 0, 0, 0]])[0, 0]
                        
                        # Рассчитываем уверенность
                        confidence = max(0, min(1, 1 - abs(predicted_price - current_price) / current_price))
                        
                        # Сигнал на покупку/продажу
                        price_change_pct = (predicted_price - current_price) / current_price
                        
                        # Покупка при прогнозе роста > 1.5%
                        if price_change_pct > 0.015 and symbol not in positions and confidence > 0.5:
                            position_size = self.calculate_position_size(current_price)
                            if position_size > 0:
                                positions[symbol] = {
                                    'size': position_size,
                                    'entry_price': current_price,
                                    'entry_date': df.index[i],
                                    'predicted_price': predicted_price,
                                    'confidence': confidence,
                                    'method': 'LSTM'
                                }
                                logger.info(f"LSTM покупка {symbol}: {position_size} акций по {current_price:.2f}₽ (прогноз: {predicted_price:.2f}₽)")
                        
                        # Продажа при прогнозе падения > 1%
                        elif price_change_pct < -0.01 and symbol in positions:
                            position = positions[symbol]
                            exit_price = current_price
                            pnl = (exit_price - position['entry_price']) * position['size']
                            
                            trades.append({
                                'entry_date': position['entry_date'],
                                'exit_date': df.index[i],
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'return_pct': (exit_price / position['entry_price'] - 1) * 100,
                                'method': 'LSTM',
                                'confidence': confidence
                            })
                            
                            current_capital += pnl
                            del positions[symbol]
                            logger.info(f"LSTM продажа {symbol}: PnL={pnl:.2f}₽")
                
                except Exception as e:
                    logger.debug(f"Ошибка LSTM для {symbol}: {e}")
                
                # Обновляем метрики
                current_equity = current_capital
                if symbol in positions:
                    position = positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_history.append({
                    'date': df.index[i],
                    'equity': current_equity
                })
            
            return self._calculate_metrics(symbol, equity_history, trades, "LSTM_Strategy")
            
        except Exception as e:
            logger.error(f"Ошибка в LSTM стратегии для {symbol}: {e}")
            return None
    
    def ensemble_ml_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Ансамблевая ML стратегия с несколькими моделями"""
        try:
            logger.info(f"Тестирование Ensemble ML стратегии для {symbol}")
            
            # Подготавливаем данные
            df = self.create_technical_features(data)
            df = df.dropna()
            
            if len(df) < 100:
                return None
            
            current_capital = self.initial_capital
            positions = {}
            trades = []
            equity_history = []
            
            # Подготавливаем признаки
            feature_columns = ['sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'rsi', 'macd', 
                             'bb_position', 'stoch_k', 'volume_ratio', 'momentum_5', 'momentum_10',
                             'volatility_ratio', 'price_change', 'high_low_ratio']
            
            for i in range(50, len(df)):
                current_price = df.iloc[i]['close']
                
                # Проверяем стоп-ордера
                if symbol in positions:
                    self.check_stop_orders(symbol, current_price, positions, trades, current_capital)
                
                try:
                    # Подготавливаем данные для обучения
                    train_data = df.iloc[max(0, i-100):i]
                    if len(train_data) < 50:
                        continue
                    
                    X_train = train_data[feature_columns].values
                    y_train = train_data['close'].shift(-1).dropna().values
                    
                    # Убираем последний элемент из X_train, так как для него нет y
                    X_train = X_train[:-1]
                    
                    if len(X_train) < 30:
                        continue
                    
                    # Нормализация
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    # Создаем ансамбль моделей
                    models = {
                        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
                        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
                        'Ridge': Ridge(alpha=1.0),
                        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
                    }
                    
                    # Обучаем модели
                    predictions = []
                    model_scores = []
                    
                    for name, model in models.items():
                        try:
                            model.fit(X_train_scaled, y_train)
                            
                            # Предсказываем на последних данных
                            last_features = df.iloc[i][feature_columns].values.reshape(1, -1)
                            last_features_scaled = scaler.transform(last_features)
                            pred = model.predict(last_features_scaled)[0]
                            predictions.append(pred)
                            
                            # Рассчитываем score модели
                            train_pred = model.predict(X_train_scaled)
                            score = 1 - mean_squared_error(y_train, train_pred) / np.var(y_train)
                            model_scores.append(max(0, score))
                            
                        except Exception as e:
                            logger.debug(f"Ошибка модели {name}: {e}")
                            predictions.append(current_price)
                            model_scores.append(0)
                    
                    if len(predictions) > 0:
                        # Взвешенное среднее предсказаний
                        weights = np.array(model_scores) / sum(model_scores) if sum(model_scores) > 0 else np.ones(len(predictions)) / len(predictions)
                        predicted_price = np.average(predictions, weights=weights)
                        
                        # Рассчитываем уверенность
                        confidence = np.mean(model_scores)
                        
                        # Сигнал на покупку/продажу
                        price_change_pct = (predicted_price - current_price) / current_price
                        
                        # Покупка при прогнозе роста > 1%
                        if price_change_pct > 0.01 and symbol not in positions and confidence > 0.3:
                            position_size = self.calculate_position_size(current_price)
                            if position_size > 0:
                                positions[symbol] = {
                                    'size': position_size,
                                    'entry_price': current_price,
                                    'entry_date': df.index[i],
                                    'predicted_price': predicted_price,
                                    'confidence': confidence,
                                    'method': 'Ensemble_ML'
                                }
                                logger.info(f"Ensemble покупка {symbol}: {position_size} акций по {current_price:.2f}₽ (прогноз: {predicted_price:.2f}₽)")
                        
                        # Продажа при прогнозе падения > 0.5%
                        elif price_change_pct < -0.005 and symbol in positions:
                            position = positions[symbol]
                            exit_price = current_price
                            pnl = (exit_price - position['entry_price']) * position['size']
                            
                            trades.append({
                                'entry_date': position['entry_date'],
                                'exit_date': df.index[i],
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'return_pct': (exit_price / position['entry_price'] - 1) * 100,
                                'method': 'Ensemble_ML',
                                'confidence': confidence
                            })
                            
                            current_capital += pnl
                            del positions[symbol]
                            logger.info(f"Ensemble продажа {symbol}: PnL={pnl:.2f}₽")
                
                except Exception as e:
                    logger.debug(f"Ошибка Ensemble ML для {symbol}: {e}")
                
                # Обновляем метрики
                current_equity = current_capital
                if symbol in positions:
                    position = positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_history.append({
                    'date': df.index[i],
                    'equity': current_equity
                })
            
            return self._calculate_metrics(symbol, equity_history, trades, "Ensemble_ML_Strategy")
            
        except Exception as e:
            logger.error(f"Ошибка в Ensemble ML стратегии для {symbol}: {e}")
            return None
    
    def sarima_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """SARIMA стратегия с учетом сезонности"""
        try:
            logger.info(f"Тестирование SARIMA стратегии для {symbol}")
            
            # Подготавливаем данные
            df = self.create_technical_features(data)
            df = df.dropna()
            
            if len(df) < 100:
                return None
            
            current_capital = self.initial_capital
            positions = {}
            trades = []
            equity_history = []
            
            for i in range(50, len(df)):
                current_price = df.iloc[i]['close']
                
                # Проверяем стоп-ордера
                if symbol in positions:
                    self.check_stop_orders(symbol, current_price, positions, trades, current_capital)
                
                try:
                    # Получаем исторические данные
                    price_series = df.iloc[i-50:i]['close'].values
                    
                    # Простая SARIMA модель
                    model = SARIMAX(price_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
                    fitted_model = model.fit(disp=False)
                    
                    # Делаем прогноз
                    forecast = fitted_model.forecast(steps=1)
                    predicted_price = forecast[0]
                    
                    # Рассчитываем уверенность
                    confidence = max(0, min(1, 1 - abs(predicted_price - current_price) / current_price))
                    
                    # Сигнал на покупку/продажу
                    price_change_pct = (predicted_price - current_price) / current_price
                    
                    # Покупка при прогнозе роста > 1.5%
                    if price_change_pct > 0.015 and symbol not in positions and confidence > 0.5:
                        position_size = self.calculate_position_size(current_price)
                        if position_size > 0:
                            positions[symbol] = {
                                'size': position_size,
                                'entry_price': current_price,
                                'entry_date': df.index[i],
                                'predicted_price': predicted_price,
                                'confidence': confidence,
                                'method': 'SARIMA'
                            }
                            logger.info(f"SARIMA покупка {symbol}: {position_size} акций по {current_price:.2f}₽ (прогноз: {predicted_price:.2f}₽)")
                    
                    # Продажа при прогнозе падения > 1%
                    elif price_change_pct < -0.01 and symbol in positions:
                        position = positions[symbol]
                        exit_price = current_price
                        pnl = (exit_price - position['entry_price']) * position['size']
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': df.index[i],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'return_pct': (exit_price / position['entry_price'] - 1) * 100,
                            'method': 'SARIMA',
                            'confidence': confidence
                        })
                        
                        current_capital += pnl
                        del positions[symbol]
                        logger.info(f"SARIMA продажа {symbol}: PnL={pnl:.2f}₽")
                
                except Exception as e:
                    logger.debug(f"Ошибка SARIMA для {symbol}: {e}")
                
                # Обновляем метрики
                current_equity = current_capital
                if symbol in positions:
                    position = positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_history.append({
                    'date': df.index[i],
                    'equity': current_equity
                })
            
            return self._calculate_metrics(symbol, equity_history, trades, "SARIMA_Strategy")
            
        except Exception as e:
            logger.error(f"Ошибка в SARIMA стратегии для {symbol}: {e}")
            return None
    
    def calculate_position_size(self, current_price: float) -> int:
        """Расчет размера позиции"""
        try:
            max_risk_amount = self.current_capital * self.max_risk_per_trade
            stop_loss_price = current_price * (1 - self.stop_loss_pct)
            risk_per_share = current_price - stop_loss_price
            
            if risk_per_share <= 0:
                return 0
            
            position_size = int(max_risk_amount / risk_per_share)
            max_position_value = self.current_capital * 0.2  # 20% от капитала
            max_position_size = int(max_position_value / current_price)
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {e}")
            return 0
    
    def check_stop_orders(self, symbol: str, current_price: float, positions: Dict, trades: List, current_capital: float):
        """Проверка стоп-ордеров"""
        try:
            if symbol not in positions:
                return
            
            position = positions[symbol]
            stop_loss_price = position['entry_price'] * (1 - self.stop_loss_pct)
            take_profit_price = position['entry_price'] * (1 + self.take_profit_pct)
            
            # Стоп-лосс
            if current_price <= stop_loss_price:
                exit_price = current_price
                pnl = (exit_price - position['entry_price']) * position['size']
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': datetime.now(),
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return_pct': (exit_price / position['entry_price'] - 1) * 100,
                    'method': position['method'],
                    'exit_reason': 'stop_loss'
                })
                
                current_capital += pnl
                del positions[symbol]
                logger.info(f"Стоп-лосс {symbol}: PnL={pnl:.2f}₽")
            
            # Тейк-профит
            elif current_price >= take_profit_price:
                exit_price = current_price
                pnl = (exit_price - position['entry_price']) * position['size']
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': datetime.now(),
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return_pct': (exit_price / position['entry_price'] - 1) * 100,
                    'method': position['method'],
                    'exit_reason': 'take_profit'
                })
                
                current_capital += pnl
                del positions[symbol]
                logger.info(f"Тейк-профит {symbol}: PnL={pnl:.2f}₽")
                
        except Exception as e:
            logger.error(f"Ошибка проверки стоп-ордеров: {e}")
    
    def _calculate_metrics(self, symbol: str, equity_history: List, trades: List, strategy_name: str) -> Dict:
        """Расчет метрик стратегии"""
        try:
            if not equity_history:
                return None
            
            equity_curve = pd.DataFrame(equity_history)
            equity_curve.set_index('date', inplace=True)
            
            # Основные метрики
            total_return = (equity_curve['equity'].iloc[-1] / self.initial_capital) - 1
            
            # Месячная доходность
            period_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            months_in_period = period_days / 30.44
            monthly_return = ((1 + total_return) ** (1 / months_in_period) - 1) * 100 if months_in_period > 0 else 0
            
            # Волатильность
            volatility = equity_curve['equity'].pct_change().std() * np.sqrt(252) * 100
            
            # Sharpe ratio
            risk_free_rate = 0.05
            excess_returns = equity_curve['equity'].pct_change().mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (equity_curve['equity'].pct_change().std() * np.sqrt(252)) if equity_curve['equity'].pct_change().std() > 0 else 0
            
            # Максимальная просадка
            rolling_max = equity_curve['equity'].expanding().max()
            drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Win rate
            if trades:
                winning_trades = [t for t in trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(trades) * 100
            else:
                win_rate = 0
            
            # Статистика по методам
            method_stats = {}
            for trade in trades:
                method = trade.get('method', 'Unknown')
                if method not in method_stats:
                    method_stats[method] = {'trades': 0, 'pnl': 0}
                method_stats[method]['trades'] += 1
                method_stats[method]['pnl'] += trade['pnl']
            
            return {
                'symbol': symbol,
                'strategy': strategy_name,
                'monthly_return': float(monthly_return),
                'total_return': float(total_return * 100),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'total_trades': int(len(trades)),
                'final_equity': float(equity_curve['equity'].iloc[-1]),
                'method_stats': method_stats,
                'period_days': int(period_days)
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик для {symbol}: {e}")
            return None
    
    def load_tbank_data(self):
        """Загрузка данных T-Bank"""
        data_dir = 'data/tbank_real'
        market_data = {}
        
        if not os.path.exists(data_dir):
            logger.error(f"Директория {data_dir} не найдена")
            return market_data
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_tbank.csv'):
                symbol_period = filename.replace('_tbank.csv', '')
                parts = symbol_period.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    period = parts[1]
                    
                    filepath = os.path.join(data_dir, filename)
                    try:
                        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
                        if not df.empty and len(df) >= 100:  # Минимум 100 дней для ML
                            key = f"{symbol}_{period}"
                            market_data[key] = df
                            logger.info(f"Загружены данные {key}: {len(df)} дней")
                    except Exception as e:
                        logger.error(f"Ошибка загрузки {filename}: {e}")
        
        return market_data
    
    def run_advanced_ml_testing(self):
        """Запуск тестирования продвинутых ML стратегий"""
        logger.info("🤖 ТЕСТИРОВАНИЕ ПРОДВИНУТЫХ ML СТРАТЕГИЙ")
        logger.info("=" * 60)
        
        # Загружаем данные
        market_data = self.load_tbank_data()
        
        if not market_data:
            logger.error("❌ Нет данных для тестирования")
            return
        
        logger.info(f"✅ Загружено {len(market_data)} наборов данных")
        
        # Определяем стратегии для тестирования
        strategies = [
            ('ARIMA_Strategy', self.arima_strategy),
            ('Ensemble_ML_Strategy', self.ensemble_ml_strategy),
            ('SARIMA_Strategy', self.sarima_strategy)
        ]
        
        if LSTM_AVAILABLE:
            strategies.append(('LSTM_Strategy', self.lstm_strategy))
        
        # Тестируем все стратегии
        all_results = {}
        
        for data_key, data in market_data.items():
            symbol = data_key.split('_')[0]
            logger.info(f"\n📈 Тестирование {data_key}...")
            
            results = {}
            
            for strategy_name, strategy_func in strategies:
                logger.info(f"  🔍 {strategy_name}...")
                result = strategy_func(symbol, data)
                if result:
                    results[strategy_name] = result
                    logger.info(f"    ✅ {result['monthly_return']:.2f}% в месяц, Sharpe: {result['sharpe_ratio']:.2f}")
                else:
                    logger.info(f"    ❌ Не удалось получить результаты")
            
            all_results[data_key] = results
        
        # Сохраняем результаты
        self._save_advanced_results(all_results)
        
        # Выводим сводку
        self._print_advanced_summary(all_results)
        
        return all_results
    
    def _save_advanced_results(self, results):
        """Сохранение результатов продвинутых стратегий"""
        output_dir = 'backtesting/results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        json_filename = os.path.join(output_dir, f'advanced_ml_strategies_{timestamp}.json')
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Результаты сохранены в {json_filename}")
    
    def _print_advanced_summary(self, results):
        """Вывод сводки продвинутых стратегий"""
        logger.info(f"\n🤖 СВОДКА ПРОДВИНУТЫХ ML СТРАТЕГИЙ:")
        
        # Собираем все результаты
        all_strategy_results = []
        for data_key, data_results in results.items():
            for strategy_name, strategy_result in data_results.items():
                all_strategy_results.append({
                    'data_key': data_key,
                    'strategy': strategy_name,
                    'monthly_return': strategy_result['monthly_return'],
                    'sharpe_ratio': strategy_result['sharpe_ratio'],
                    'max_drawdown': strategy_result['max_drawdown'],
                    'total_trades': strategy_result['total_trades'],
                    'method_stats': strategy_result.get('method_stats', {})
                })
        
        # Сортируем по месячной доходности
        all_strategy_results.sort(key=lambda x: x['monthly_return'], reverse=True)
        
        logger.info(f"\n🏆 ТОП-10 ПРОДВИНУТЫХ ML СТРАТЕГИЙ:")
        for i, result in enumerate(all_strategy_results[:10]):
            logger.info(f"  {i+1}. {result['data_key']} - {result['strategy']}: {result['monthly_return']:.2f}% в месяц")
            logger.info(f"      Sharpe: {result['sharpe_ratio']:.2f}, Просадка: {result['max_drawdown']:.2f}%, Сделок: {result['total_trades']}")

def main():
    """Основная функция"""
    tester = AdvancedMLStrategies()
    results = tester.run_advanced_ml_testing()
    
    if results:
        logger.info(f"\n🤖 ТЕСТИРОВАНИЕ ПРОДВИНУТЫХ ML СТРАТЕГИЙ ЗАВЕРШЕНО!")
    else:
        logger.info(f"\n❌ ТЕСТИРОВАНИЕ НЕ УДАЛОСЬ")

if __name__ == "__main__":
    main()
