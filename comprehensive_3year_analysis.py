#!/usr/bin/env python3
"""
Комплексный анализ за 3 года: загрузка минутных данных, обучение моделей и тестирование стратегий
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML импорты
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('3year_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataDownloader:
    """Класс для загрузки минутных исторических данных с MOEX"""
    
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss/"
        self.data_dir = "data/3year_minute_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_instrument_data(self, symbol: str, years: int = 3) -> pd.DataFrame:
        """Загрузка минутных данных за указанное количество лет"""
        logger.info(f"📥 Загружаем минутные данные для {symbol} за {years} лет...")
        
        all_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Разбиваем на части по 30 дней (из-за лимита 500 записей)
        current_date = start_date
        batch_size = 30  # дней
        
        while current_date < end_date:
            batch_end = min(current_date + timedelta(days=batch_size), end_date)
            
            try:
                url = f"{self.base_url}engines/stock/markets/shares/boards/TQBR/securities/{symbol}/candles.json"
                params = {
                    'from': current_date.strftime('%Y-%m-%d'),
                    'till': batch_end.strftime('%Y-%m-%d'),
                    'interval': 1  # 1 минута
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    candles = data['candles']['data']
                    
                    if candles:
                        df_batch = pd.DataFrame(candles, columns=[
                            'open', 'close', 'high', 'low', 'value', 'volume', 'begin', 'end'
                        ])
                        df_batch['begin'] = pd.to_datetime(df_batch['begin'])
                        df_batch['end'] = pd.to_datetime(df_batch['end'])
                        df_batch['symbol'] = symbol
                        
                        all_data.append(df_batch)
                        logger.info(f"  ✅ {symbol}: {len(candles)} записей за {current_date.strftime('%Y-%m-%d')} - {batch_end.strftime('%Y-%m-%d')}")
                    else:
                        logger.warning(f"  ⚠️ {symbol}: Нет данных за {current_date.strftime('%Y-%m-%d')} - {batch_end.strftime('%Y-%m-%d')}")
                else:
                    logger.error(f"  ❌ {symbol}: Ошибка {response.status_code} за {current_date.strftime('%Y-%m-%d')}")
                
                current_date = batch_end
                time.sleep(0.3)  # Пауза между запросами
                
            except Exception as e:
                logger.error(f"  ❌ {symbol}: Ошибка за {current_date.strftime('%Y-%m-%d')}: {e}")
                current_date = batch_end
                time.sleep(1)
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('begin').reset_index(drop=True)
            
            # Удаляем дубликаты
            df = df.drop_duplicates(subset=['begin']).reset_index(drop=True)
            
            # Сохраняем данные
            filename = f"{self.data_dir}/{symbol}_3year_minute.csv"
            df.to_csv(filename, index=False)
            logger.info(f"💾 {symbol}: Сохранено {len(df)} записей в {filename}")
            
            return df
        else:
            logger.error(f"❌ {symbol}: Не удалось загрузить данные")
            return pd.DataFrame()
    
    def download_all_instruments(self, symbols: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
        """Загрузка данных для всех инструментов"""
        logger.info(f"🚀 Начинаем загрузку минутных данных для {len(symbols)} инструментов за {years} лет")
        
        all_data = {}
        
        for symbol in symbols:
            try:
                df = self.download_instrument_data(symbol, years)
                if not df.empty:
                    all_data[symbol] = df
                else:
                    logger.error(f"❌ Не удалось загрузить данные для {symbol}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {symbol}: {e}")
        
        logger.info(f"✅ Загружено данных для {len(all_data)} инструментов")
        return all_data

class TechnicalIndicators:
    """Расширенный класс для расчета технических индикаторов"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Простая скользящая средняя"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Экспоненциальная скользящая средняя"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Индекс относительной силы"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD индикатор"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Полосы Боллинджера"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    @staticmethod
    def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание всех технических индикаторов"""
        df = df.copy()
        
        # Основные скользящие средние
        for window in [5, 10, 20, 50, 100]:
            df[f'sma_{window}'] = TechnicalIndicators.sma(df['close'], window)
            df[f'ema_{window}'] = TechnicalIndicators.ema(df['close'], window)
        
        # RSI с разными периодами
        for window in [14, 21]:
            df[f'rsi_{window}'] = TechnicalIndicators.rsi(df['close'], window)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Полосы Боллинджера
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Stochastic
        k_percent, d_percent = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        # Williams %R
        df['williams_r'] = TechnicalIndicators.williams_r(df['high'], df['low'], df['close'])
        
        # CCI
        df['cci'] = TechnicalIndicators.cci(df['high'], df['low'], df['close'])
        
        # Дополнительные признаки
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        
        # Волатильность
        df['volatility_5'] = df['close'].rolling(5).std()
        df['volatility_20'] = df['close'].rolling(20).std()
        
        # Временные признаки
        df['hour'] = df['begin'].dt.hour
        df['day_of_week'] = df['begin'].dt.dayofweek
        df['month'] = df['begin'].dt.month
        df['is_market_open'] = ((df['hour'] >= 10) & (df['hour'] < 18)).astype(int)
        
        # Объемные индикаторы
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['volume']
        
        return df

class AdvancedMLTrainer:
    """Продвинутый класс для обучения ML моделей"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_importance = {}
        self.model_scores = {}
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'future_return', 
                    lookback: int = 60, forecast_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Подготовка данных для обучения с расширенными признаками"""
        # Создаем целевую переменную
        df['future_price'] = df['close'].shift(-forecast_horizon)
        df['future_return'] = (df['future_price'] / df['close'] - 1) * 100
        
        # Убираем NaN
        df = df.dropna()
        
        # Выбираем все числовые признаки
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in ['future_price', 'future_return', 'value']]
        
        # Создаем скользящие окна
        X = []
        y = []
        
        for i in range(lookback, len(df) - forecast_horizon):
            window_data = df[feature_columns].iloc[i-lookback:i].values
            target_value = df[target_column].iloc[i]
            
            if not np.isnan(target_value):
                X.append(window_data.flatten())
                y.append(target_value)
        
        return np.array(X), np.array(y), feature_columns
    
    def train_models(self, X: np.ndarray, y: np.ndarray, symbol: str, feature_columns: List[str]):
        """Обучение различных ML моделей с оптимизацией"""
        logger.info(f"🤖 Обучаем продвинутые модели для {symbol}...")
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Выбор признаков
        feature_selector = SelectKBest(score_func=f_regression, k=min(50, X_train_scaled.shape[1]))
        X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = feature_selector.transform(X_test_scaled)
        
        # Модели для обучения
        models_config = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models_config.items():
            try:
                logger.info(f"  🔄 Обучаем {name}...")
                
                # Обучение
                if name in ['ridge', 'svr', 'linear_regression']:
                    model.fit(X_train_selected, y_train)
                    y_pred = model.predict(X_test_selected)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Оценка
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Кросс-валидация
                if name in ['ridge', 'svr', 'linear_regression']:
                    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                logger.info(f"    ✅ {name}: MSE={mse:.4f}, R²={r2:.4f}, MAE={mae:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"    ❌ Ошибка обучения {name}: {e}")
        
        # Сохраняем результаты
        self.models[symbol] = results
        self.scalers[symbol] = scaler
        self.feature_selectors[symbol] = feature_selector
        
        return results

class UnsupervisedLearningAgent:
    """Агент для обучения без учителя"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        self.pca = PCA(n_components=10)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Подготовка признаков для обучения без учителя"""
        # Выбираем ключевые признаки
        feature_columns = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_width', 'atr',
            'stoch_k', 'williams_r', 'cci', 'volatility_20', 'volume_ratio',
            'price_change', 'high_low_ratio', 'body_size'
        ]
        
        # Фильтруем доступные колонки
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_columns:
            return np.array([])
        
        features = df[available_columns].fillna(0).values
        return features
    
    def train_unsupervised_models(self, df: pd.DataFrame):
        """Обучение моделей без учителя"""
        logger.info("🤖 Обучение моделей без учителя...")
        
        features = self.prepare_features(df)
        if len(features) < 100:
            logger.warning("⚠️ Недостаточно данных для обучения без учителя")
            return
        
        # Нормализация
        features_scaled = self.scaler.fit_transform(features)
        
        # PCA для снижения размерности
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Обучение детектора аномалий
        self.anomaly_detector.fit(features_pca)
        
        # Обучение кластеризации
        self.clusterer.fit(features_pca)
        
        self.is_trained = True
        logger.info("✅ Модели без учителя обучены")
    
    def detect_anomalies(self, df: pd.DataFrame) -> np.ndarray:
        """Детекция аномалий"""
        if not self.is_trained:
            return np.array([])
        
        features = self.prepare_features(df)
        if len(features) == 0:
            return np.array([])
        
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        anomalies = self.anomaly_detector.predict(features_pca)
        return anomalies
    
    def get_market_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Определение рыночных режимов"""
        if not self.is_trained:
            return np.array([])
        
        features = self.prepare_features(df)
        if len(features) == 0:
            return np.array([])
        
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        clusters = self.clusterer.predict(features_pca)
        return clusters

class AdvancedStrategyTester:
    """Продвинутый класс для тестирования торговых стратегий"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
    
    def test_ml_strategy(self, df: pd.DataFrame, models: Dict, symbol: str, 
                        unsupervised_agent: UnsupervisedLearningAgent) -> Dict:
        """Тестирование продвинутой ML стратегии"""
        logger.info(f"📊 Тестируем продвинутую ML стратегию для {symbol}...")
        
        capital = self.initial_capital
        position = 0
        trades = []
        equity_history = []
        
        # Параметры стратегии
        lookback = 60
        forecast_horizon = 5
        confidence_threshold = 0.6  # Повышенный порог уверенности
        
        for i in range(lookback, len(df) - forecast_horizon):
            current_price = df['close'].iloc[i]
            
            # Подготавливаем данные для предсказания
            feature_columns = [col for col in df.columns if col not in ['begin', 'end', 'symbol', 'future_price', 'future_return', 'value']]
            window_data = df[feature_columns].iloc[i-lookback:i].values.flatten()
            
            # Получаем предсказания от всех моделей
            predictions = []
            model_weights = []
            
            for model_name, model_data in models.items():
                try:
                    if model_name in ['ridge', 'svr', 'linear_regression']:
                        # Нужна нормализация и выбор признаков
                        window_scaled = models['scaler'].transform([window_data])
                        window_selected = models['feature_selector'].transform(window_scaled)
                        pred = model_data['model'].predict(window_selected)[0]
                    else:
                        pred = model_data['model'].predict([window_data])[0]
                    
                    predictions.append(pred)
                    # Вес модели основан на R²
                    model_weights.append(max(0, model_data['r2']))
                    
                except:
                    continue
            
            if not predictions:
                continue
            
            # Взвешенное усреднение предсказаний
            if sum(model_weights) > 0:
                avg_prediction = np.average(predictions, weights=model_weights)
                confidence = 1.0 - (np.std(predictions) / (abs(avg_prediction) + 1e-6))
            else:
                avg_prediction = np.mean(predictions)
                confidence = 0.5
            
            # Анализ аномалий и режимов
            anomaly_signal = 0.0
            regime_signal = 0.0
            
            if unsupervised_agent.is_trained:
                current_data = df.iloc[max(0, i-100):i+1]
                anomalies = unsupervised_agent.detect_anomalies(current_data)
                regimes = unsupervised_agent.get_market_regimes(current_data)
                
                if len(anomalies) > 0 and anomalies[-1] == -1:
                    anomaly_signal = 0.3  # Аномалия может указывать на разворот
                
                if len(regimes) > 0:
                    current_regime = regimes[-1]
                    regime_signal = (current_regime - 2) * 0.1  # Центрируем вокруг 0
            
            # Объединяем сигналы
            final_signal = avg_prediction + anomaly_signal + regime_signal
            
            # Логика торговли с улучшенными условиями
            if final_signal > 0.5 and position == 0 and confidence > confidence_threshold:
                position = capital / current_price
                capital = 0
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'time': df['begin'].iloc[i],
                    'prediction': avg_prediction,
                    'confidence': confidence,
                    'signal': final_signal
                })
            elif final_signal < -0.5 and position > 0 and confidence > confidence_threshold:
                capital = position * current_price
                position = 0
                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'time': df['begin'].iloc[i],
                    'prediction': avg_prediction,
                    'confidence': confidence,
                    'signal': final_signal
                })
            
            # Текущая стоимость портфеля
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_history.append({
                'time': df['begin'].iloc[i],
                'equity': current_equity,
                'price': current_price
            })
        
        # Закрываем позицию в конце
        if position > 0:
            final_price = df['close'].iloc[-1]
            capital = position * final_price
            trades.append({
                'type': 'sell',
                'price': final_price,
                'time': df['begin'].iloc[-1],
                'prediction': 0,
                'confidence': 0,
                'signal': 0
            })
        
        # Расчет метрик
        final_equity = capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        # Максимальная просадка
        equity_series = pd.Series([h['equity'] for h in equity_history])
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Волатильность
        returns = equity_series.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252 * 24 * 60) * 100  # Годовая волатильность
        
        # Sharpe ratio
        risk_free_rate = 0.05  # 5% годовых
        excess_returns = returns - risk_free_rate / (252 * 24 * 60)
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0
        
        # Win rate
        if len(trades) >= 2:
            buy_trades = [t for t in trades if t['type'] == 'buy']
            sell_trades = [t for t in trades if t['type'] == 'sell']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                profitable_trades = 0
                total_trades = min(len(buy_trades), len(sell_trades))
                
                for i in range(total_trades):
                    if i < len(sell_trades):
                        profit = (sell_trades[i]['price'] - buy_trades[i]['price']) / buy_trades[i]['price']
                        if profit > 0:
                            profitable_trades += 1
                
                win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        # Дополнительные метрики
        avg_trade_duration = 0
        if len(trades) > 0:
            trade_durations = []
            for i in range(0, len(trades)-1, 2):
                if i+1 < len(trades):
                    duration = (trades[i+1]['time'] - trades[i]['time']).total_seconds() / 3600  # в часах
                    trade_durations.append(duration)
            avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        result = {
            'symbol': symbol,
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_trade_duration_hours': avg_trade_duration,
            'trades': trades,
            'equity_history': equity_history
        }
        
        logger.info(f"  ✅ {symbol}: Доходность={total_return:.2f}%, Просадка={max_drawdown:.2f}%, Sharpe={sharpe_ratio:.2f}, Win Rate={win_rate:.1f}%")
        
        return result
    
    def test_buy_hold_strategy(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Тестирование стратегии Buy & Hold"""
        logger.info(f"📈 Тестируем Buy & Hold для {symbol}...")
        
        initial_price = df['close'].iloc[0]
        final_price = df['close'].iloc[-1]
        
        total_return = (final_price - initial_price) / initial_price * 100
        
        # Максимальная просадка
        rolling_max = df['close'].expanding().max()
        drawdown = (df['close'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Волатильность
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252 * 24 * 60) * 100
        
        # Sharpe ratio
        risk_free_rate = 0.05
        excess_returns = returns - risk_free_rate / (252 * 24 * 60)
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0
        
        result = {
            'symbol': symbol,
            'strategy': 'buy_hold',
            'initial_price': initial_price,
            'final_price': final_price,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': 100 if total_return > 0 else 0,
            'total_trades': 1
        }
        
        logger.info(f"  ✅ {symbol} Buy & Hold: Доходность={total_return:.2f}%, Просадка={max_drawdown:.2f}%")
        
        return result

def main():
    """Основная функция"""
    logger.info("🚀 ЗАПУСК КОМПЛЕКСНОГО АНАЛИЗА ЗА 3 ГОДА (МИНУТНЫЕ ДАННЫЕ)")
    logger.info("=" * 70)
    
    # Инструменты для анализа
    symbols = ['GAZP', 'SBER', 'PIKK', 'IRAO', 'SGZH']
    
    # 1. Загрузка данных
    logger.info("📥 ЭТАП 1: Загрузка минутных исторических данных")
    downloader = DataDownloader()
    all_data = downloader.download_all_instruments(symbols, years=3)
    
    if not all_data:
        logger.error("❌ Не удалось загрузить данные. Завершение работы.")
        return
    
    # 2. Подготовка данных и обучение моделей
    logger.info("\n🤖 ЭТАП 2: Обучение продвинутых ML моделей")
    trainer = AdvancedMLTrainer()
    indicators = TechnicalIndicators()
    unsupervised_agent = UnsupervisedLearningAgent()
    
    trained_models = {}
    
    for symbol, df in all_data.items():
        logger.info(f"📊 Обрабатываем {symbol}...")
        
        # Создаем технические индикаторы
        df_with_features = indicators.create_comprehensive_features(df)
        
        # Подготавливаем данные для обучения
        X, y, feature_columns = trainer.prepare_data(df_with_features)
        
        if len(X) > 200:  # Минимум данных для обучения
            # Обучаем модели
            models = trainer.train_models(X, y, symbol, feature_columns)
            trained_models[symbol] = {
                'models': models,
                'data': df_with_features
            }
        else:
            logger.warning(f"⚠️ {symbol}: Недостаточно данных для обучения ({len(X)} записей)")
    
    # Обучение моделей без учителя
    if trained_models:
        all_training_data = []
        for model_data in trained_models.values():
            all_training_data.append(model_data['data'])
        
        combined_df = pd.concat(all_training_data, ignore_index=True)
        unsupervised_agent.train_unsupervised_models(combined_df)
    
    # 3. Тестирование стратегий
    logger.info("\n📊 ЭТАП 3: Тестирование торговых стратегий")
    tester = AdvancedStrategyTester()
    
    all_results = {}
    
    for symbol, model_data in trained_models.items():
        logger.info(f"🧪 Тестируем стратегии для {symbol}...")
        
        df = model_data['data']
        models = model_data['models']
        
        # Тестируем ML стратегию
        ml_result = tester.test_ml_strategy(df, models, symbol, unsupervised_agent)
        
        # Тестируем Buy & Hold
        bh_result = tester.test_buy_hold_strategy(df, symbol)
        
        all_results[symbol] = {
            'ml_strategy': ml_result,
            'buy_hold': bh_result
        }
    
    # 4. Анализ результатов
    logger.info("\n📈 ЭТАП 4: Анализ результатов")
    
    # Создаем сводную таблицу
    summary_data = []
    
    for symbol, results in all_results.items():
        ml = results['ml_strategy']
        bh = results['buy_hold']
        
        summary_data.append({
            'Symbol': symbol,
            'ML_Return': f"{ml['total_return']:.2f}%",
            'ML_Drawdown': f"{ml['max_drawdown']:.2f}%",
            'ML_Sharpe': f"{ml['sharpe_ratio']:.2f}",
            'ML_Win_Rate': f"{ml['win_rate']:.1f}%",
            'ML_Trades': ml['total_trades'],
            'ML_Avg_Duration': f"{ml['avg_trade_duration_hours']:.1f}h",
            'BH_Return': f"{bh['total_return']:.2f}%",
            'BH_Drawdown': f"{bh['max_drawdown']:.2f}%",
            'BH_Sharpe': f"{bh['sharpe_ratio']:.2f}",
            'Outperformance': f"{ml['total_return'] - bh['total_return']:.2f}%"
        })
    
    # Сохраняем результаты
    results_df = pd.DataFrame(summary_data)
    results_df.to_csv('3year_analysis_results.csv', index=False)
    
    # Сохраняем детальные результаты
    def clean_for_json(obj):
        """Очистка объекта от циклических ссылок и numpy типов"""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() 
                   if k not in ['model', 'scaler', 'feature_selector', 'anomaly_detector', 'clusterer']}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        else:
            return obj
    
    with open('3year_detailed_results.json', 'w', encoding='utf-8') as f:
        clean_results = clean_for_json(all_results)
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # Выводим сводку
    logger.info("\n📋 СВОДКА РЕЗУЛЬТАТОВ (3 ГОДА, МИНУТНЫЕ ДАННЫЕ):")
    logger.info("=" * 70)
    print(results_df.to_string(index=False))
    
    # Статистика
    ml_returns = [float(r['ML_Return'].replace('%', '')) for r in summary_data]
    bh_returns = [float(r['BH_Return'].replace('%', '')) for r in summary_data]
    ml_sharpe = [float(r['ML_Sharpe']) for r in summary_data]
    ml_win_rates = [float(r['ML_Win_Rate'].replace('%', '')) for r in summary_data]
    
    logger.info(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    logger.info(f"  Средняя доходность ML: {np.mean(ml_returns):.2f}%")
    logger.info(f"  Средняя доходность Buy & Hold: {np.mean(bh_returns):.2f}%")
    logger.info(f"  Превышение ML над Buy & Hold: {np.mean(ml_returns) - np.mean(bh_returns):.2f}%")
    logger.info(f"  Средний Sharpe ML: {np.mean(ml_sharpe):.2f}")
    logger.info(f"  Средний Win Rate ML: {np.mean(ml_win_rates):.1f}%")
    
    # Лучшие и худшие результаты
    best_return_idx = np.argmax(ml_returns)
    worst_return_idx = np.argmin(ml_returns)
    
    logger.info(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
    logger.info(f"  Лучшая доходность: {summary_data[best_return_idx]['Symbol']} ({summary_data[best_return_idx]['ML_Return']})")
    logger.info(f"  Лучший Sharpe: {summary_data[np.argmax(ml_sharpe)]['Symbol']} ({summary_data[np.argmax(ml_sharpe)]['ML_Sharpe']})")
    logger.info(f"  Лучший Win Rate: {summary_data[np.argmax(ml_win_rates)]['Symbol']} ({summary_data[np.argmax(ml_win_rates)]['ML_Win_Rate']})")
    
    logger.info(f"\n💾 Результаты сохранены:")
    logger.info(f"  📄 Сводка: 3year_analysis_results.csv")
    logger.info(f"  📄 Детали: 3year_detailed_results.json")
    logger.info(f"  📄 Логи: 3year_analysis.log")
    
    logger.info("\n✅ АНАЛИЗ ЗАВЕРШЕН!")

if __name__ == "__main__":
    main()
