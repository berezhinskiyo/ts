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
import asyncio

# Импорт системы уведомлений
try:
    from telegram_notifications import TradingNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("Telegram notifications не доступны. Установите aiohttp для поддержки уведомлений.")

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

class IndicatorOptimizer:
    """Оптимизатор технических индикаторов"""
    
    def __init__(self):
        self.performance_scores = {}
        self.correlation_matrix = None
        self.selected_indicators = []
    
    def evaluate_indicators(self, data: pd.DataFrame, target_column: str = 'close') -> Dict[str, float]:
        """Оценка эффективности индикаторов"""
        scores = {}
        
        # Создаем все возможные индикаторы
        all_indicators = self.get_all_available_indicators()
        df_with_indicators = self.create_technical_features(data, all_indicators)
        
        # Удаляем NaN значения
        df_clean = df_with_indicators.dropna()
        
        if len(df_clean) < 50:
            return scores
        
        # Рассчитываем корреляцию с целевой переменной
        target = df_clean[target_column].pct_change().shift(-1).dropna()
        
        for col in df_clean.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue
                
            try:
                # Корреляция с будущим изменением цены
                correlation = abs(df_clean[col].corr(target))
                
                # Информационный коэффициент (IC)
                ic = correlation
                
                # Коэффициент Шарпа для сигналов
                signals = np.where(df_clean[col] > df_clean[col].rolling(20).mean(), 1, -1)
                returns = signals * target
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
                
                # Комбинированная оценка
                score = 0.4 * ic + 0.3 * abs(sharpe) + 0.3 * correlation
                scores[col] = score
                
            except Exception as e:
                continue
        
        return scores
    
    def select_best_indicators(self, scores: Dict[str, float], top_n: int = 20, 
                              min_correlation: float = 0.1) -> List[str]:
        """Выбор лучших индикаторов"""
        # Фильтруем по минимальной корреляции
        filtered_scores = {k: v for k, v in scores.items() if v >= min_correlation}
        
        # Сортируем по убыванию
        sorted_indicators = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Выбираем топ-N
        selected = [indicator for indicator, score in sorted_indicators[:top_n]]
        
        # Убираем коррелированные индикаторы
        final_selected = self._remove_correlated_indicators(selected, min_correlation=0.8)
        
        return final_selected[:top_n]
    
    def _remove_correlated_indicators(self, indicators: List[str], min_correlation: float = 0.8) -> List[str]:
        """Удаление коррелированных индикаторов"""
        if len(indicators) <= 1:
            return indicators
        
        # Создаем тестовые данные для расчета корреляции
        test_data = pd.DataFrame({
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        })
        
        try:
            df_with_indicators = self.create_technical_features(test_data, indicators)
            df_clean = df_with_indicators.dropna()
            
            if len(df_clean) < 10:
                return indicators
            
            # Рассчитываем корреляционную матрицу
            corr_matrix = df_clean[indicators].corr().abs()
            
            # Удаляем высоко коррелированные
            to_remove = set()
            for i in range(len(indicators)):
                for j in range(i+1, len(indicators)):
                    if corr_matrix.iloc[i, j] > min_correlation:
                        # Удаляем тот, у которого меньше индекс (приоритет)
                        to_remove.add(indicators[j])
            
            return [ind for ind in indicators if ind not in to_remove]
            
        except Exception:
            return indicators
    
    def get_all_available_indicators(self) -> List[str]:
        """Получить список всех доступных индикаторов"""
        return [
            # Moving Averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
            
            # RSI
            'rsi_14', 'rsi_21', 'rsi_30',
            
            # MACD
            'macd_12_26_9', 'macd_5_35_5', 'macd_8_21_5',
            
            # Bollinger Bands
            'bb_20_2', 'bb_20_2.5', 'bb_20_3', 'bb_30_2', 'bb_50_2',
            
            # Stochastic
            'stoch_14_3', 'stoch_21_5', 'stoch_30_10',
            
            # Williams %R
            'williams_14', 'williams_21', 'williams_30',
            
            # CCI
            'cci_20', 'cci_30', 'cci_50',
            
            # ADX
            'adx_14', 'adx_21', 'adx_30',
            
            # Volume
            'volume_sma_10', 'volume_sma_20', 'volume_sma_50', 'obv', 'vpt',
            
            # Momentum
            'momentum_1', 'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
            
            # Volatility
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_30', 'volatility_50',
            'atr_14', 'atr_21', 'atr_30',
            
            # Patterns
            'pattern',
            
            # Trend
            'psar', 'ichimoku'
        ]
    
    def create_technical_features(self, data: pd.DataFrame, selected_indicators: List[str] = None) -> pd.DataFrame:
        """Создание технических индикаторов с возможностью выбора"""
        try:
            df = data.copy()
            
            # Если не указаны индикаторы, используем все
            if selected_indicators is None:
                selected_indicators = self.get_all_available_indicators()
            
            # Простые скользящие средние
            if any('sma' in ind for ind in selected_indicators):
                for period in [5, 10, 20, 50, 100, 200]:
                    if f'sma_{period}' in selected_indicators:
                        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Экспоненциальные скользящие средние
            if any('ema' in ind for ind in selected_indicators):
                for period in [5, 10, 20, 50, 100]:
                    if f'ema_{period}' in selected_indicators:
                        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI (разные периоды)
            if any('rsi' in ind for ind in selected_indicators):
                for period in [14, 21, 30]:
                    if f'rsi_{period}' in selected_indicators:
                        delta = df['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                        rs = gain / loss
                        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD (разные параметры)
            if any('macd' in ind for ind in selected_indicators):
                for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 21, 5)]:
                    if f'macd_{fast}_{slow}_{signal}' in selected_indicators:
                        ema_fast = df['close'].ewm(span=fast).mean()
                        ema_slow = df['close'].ewm(span=slow).mean()
                        df[f'macd_{fast}_{slow}_{signal}'] = ema_fast - ema_slow
                        df[f'macd_signal_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}_{signal}'].ewm(span=signal).mean()
                        df[f'macd_hist_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}_{signal}'] - df[f'macd_signal_{fast}_{slow}_{signal}']
            
            # Bollinger Bands (разные периоды и стандартные отклонения)
            if any('bb' in ind for ind in selected_indicators):
                for period in [20, 30, 50]:
                    for std_dev in [2, 2.5, 3]:
                        if f'bb_{period}_{std_dev}' in selected_indicators:
                            df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
                            bb_std = df['close'].rolling(window=period).std()
                            df[f'bb_upper_{period}_{std_dev}'] = df[f'bb_middle_{period}'] + (bb_std * std_dev)
                            df[f'bb_lower_{period}_{std_dev}'] = df[f'bb_middle_{period}'] - (bb_std * std_dev)
                            df[f'bb_width_{period}_{std_dev}'] = (df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}']) / df[f'bb_middle_{period}']
                            df[f'bb_position_{period}_{std_dev}'] = (df['close'] - df[f'bb_lower_{period}_{std_dev}']) / (df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}'])
            
            # Stochastic (разные периоды)
            if any('stoch' in ind for ind in selected_indicators):
                for k_period, d_period in [(14, 3), (21, 5), (30, 10)]:
                    if f'stoch_{k_period}_{d_period}' in selected_indicators:
                        low_k = df['low'].rolling(window=k_period).min()
                        high_k = df['high'].rolling(window=k_period).max()
                        df[f'stoch_k_{k_period}'] = 100 * ((df['close'] - low_k) / (high_k - low_k))
                        df[f'stoch_d_{k_period}_{d_period}'] = df[f'stoch_k_{k_period}'].rolling(window=d_period).mean()
            
            # Williams %R
            if any('williams' in ind for ind in selected_indicators):
                for period in [14, 21, 30]:
                    if f'williams_{period}' in selected_indicators:
                        high_period = df['high'].rolling(window=period).max()
                        low_period = df['low'].rolling(window=period).min()
                        df[f'williams_{period}'] = -100 * ((high_period - df['close']) / (high_period - low_period))
            
            # CCI (Commodity Channel Index)
            if any('cci' in ind for ind in selected_indicators):
                for period in [20, 30, 50]:
                    if f'cci_{period}' in selected_indicators:
                        typical_price = (df['high'] + df['low'] + df['close']) / 3
                        sma_tp = typical_price.rolling(window=period).mean()
                        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
                        df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # ADX (Average Directional Index)
            if any('adx' in ind for ind in selected_indicators):
                for period in [14, 21, 30]:
                    if f'adx_{period}' in selected_indicators:
                        high_diff = df['high'].diff()
                        low_diff = df['low'].diff()
                        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
                        minus_dm = -low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
                        
                        atr = self._calculate_atr(df, period)
                        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
                        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
                        
                        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                        df[f'adx_{period}'] = dx.rolling(window=period).mean()
            
            # Volume indicators (расширенные)
            if any('volume' in ind for ind in selected_indicators):
                for period in [10, 20, 50]:
                    if f'volume_sma_{period}' in selected_indicators:
                        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
                
                # OBV (On Balance Volume)
                if 'obv' in selected_indicators:
                    df['obv'] = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, 
                                                         np.where(df['close'] < df['close'].shift(1), -1, 0))).cumsum()
                
                # Volume Price Trend
                if 'vpt' in selected_indicators:
                    df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
            
            # Price momentum (расширенные)
            if any('momentum' in ind for ind in selected_indicators):
                for period in [1, 3, 5, 10, 20, 50]:
                    if f'momentum_{period}' in selected_indicators:
                        df[f'momentum_{period}'] = df['close'].pct_change(period)
                        df[f'momentum_abs_{period}'] = df[f'momentum_{period}'].abs()
            
            # Volatility indicators (расширенные)
            if any('volatility' in ind for ind in selected_indicators):
                for period in [5, 10, 20, 30, 50]:
                    if f'volatility_{period}' in selected_indicators:
                        df[f'volatility_{period}'] = df['close'].rolling(window=period).std()
                        df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(window=period*2).mean()
                
                # ATR (Average True Range)
                for period in [14, 21, 30]:
                    if f'atr_{period}' in selected_indicators:
                        df[f'atr_{period}'] = self._calculate_atr(df, period)
                        df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
            
            # Price patterns (расширенные)
            if any('pattern' in ind for ind in selected_indicators):
                df['price_change'] = df['close'].pct_change()
                df['price_change_abs'] = df['price_change'].abs()
                df['high_low_ratio'] = df['high'] / df['low']
                df['close_open_ratio'] = df['close'] / df['open']
                df['body_size'] = abs(df['close'] - df['open']) / df['open']
                df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
                df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
            
            # Trend indicators
            if any('trend' in ind for ind in selected_indicators):
                # Parabolic SAR
                if 'psar' in selected_indicators:
                    df['psar'] = self._calculate_psar(df)
                
                # Ichimoku Cloud
                if 'ichimoku' in selected_indicators:
                    df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
                    df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
                    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
                    df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
                    df['chikou_span'] = df['close'].shift(-26)
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания технических индикаторов: {e}")
            return data
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Расчет Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_psar(self, df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, af_maximum: float = 0.2) -> pd.Series:
        """Расчет Parabolic SAR"""
        psar = pd.Series(index=df.index, dtype=float)
        af = af_start
        ep = df['low'].iloc[0]
        trend = 1
        
        for i in range(1, len(df)):
            if trend == 1:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if df['low'].iloc[i] < psar.iloc[i]:
                    trend = -1
                    psar.iloc[i] = ep
                    ep = df['high'].iloc[i]
                    af = af_start
                else:
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af = min(af + af_increment, af_maximum)
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if df['high'].iloc[i] > psar.iloc[i]:
                    trend = 1
                    psar.iloc[i] = ep
                    ep = df['low'].iloc[i]
                    af = af_start
                else:
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af = min(af + af_increment, af_maximum)
        
        return psar

class AdvancedMLStrategies:
    """Продвинутые ML стратегии для торговли"""
    
    def __init__(self, initial_capital=100000, optimize_indicators=True, max_indicators=20, 
                 enable_telegram=True):
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
        
        # Оптимизатор индикаторов
        self.optimize_indicators = optimize_indicators
        self.max_indicators = max_indicators
        self.indicator_optimizer = IndicatorOptimizer() if optimize_indicators else None
        self.optimized_indicators = None
        self.indicator_scores = {}
        
        # Система уведомлений
        self.enable_telegram = enable_telegram and TELEGRAM_AVAILABLE
        self.telegram_notifier = None
        if self.enable_telegram:
            try:
                self.telegram_notifier = TradingNotifier()
                logger.info("Telegram уведомления включены")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать Telegram уведомления: {e}")
                self.enable_telegram = False
        
    def create_technical_features(self, data: pd.DataFrame, selected_indicators: List[str] = None) -> pd.DataFrame:
        """Создание технических индикаторов с возможностью выбора"""
        try:
            df = data.copy()
            
            # Если не указаны индикаторы, используем все
            if selected_indicators is None:
                selected_indicators = self.get_all_available_indicators()
            
            # Простые скользящие средние
            if any('sma' in ind for ind in selected_indicators):
                for period in [5, 10, 20, 50, 100, 200]:
                    if f'sma_{period}' in selected_indicators:
                        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Экспоненциальные скользящие средние
            if any('ema' in ind for ind in selected_indicators):
                for period in [5, 10, 20, 50, 100]:
                    if f'ema_{period}' in selected_indicators:
                        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI (разные периоды)
            if any('rsi' in ind for ind in selected_indicators):
                for period in [14, 21, 30]:
                    if f'rsi_{period}' in selected_indicators:
                        delta = df['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                        rs = gain / loss
                        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD (разные параметры)
            if any('macd' in ind for ind in selected_indicators):
                for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 21, 5)]:
                    if f'macd_{fast}_{slow}_{signal}' in selected_indicators:
                        ema_fast = df['close'].ewm(span=fast).mean()
                        ema_slow = df['close'].ewm(span=slow).mean()
                        df[f'macd_{fast}_{slow}_{signal}'] = ema_fast - ema_slow
                        df[f'macd_signal_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}_{signal}'].ewm(span=signal).mean()
                        df[f'macd_hist_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}_{signal}'] - df[f'macd_signal_{fast}_{slow}_{signal}']
            
            # Bollinger Bands (разные периоды и стандартные отклонения)
            if any('bb' in ind for ind in selected_indicators):
                for period in [20, 30, 50]:
                    for std_dev in [2, 2.5, 3]:
                        if f'bb_{period}_{std_dev}' in selected_indicators:
                            df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
                            bb_std = df['close'].rolling(window=period).std()
                            df[f'bb_upper_{period}_{std_dev}'] = df[f'bb_middle_{period}'] + (bb_std * std_dev)
                            df[f'bb_lower_{period}_{std_dev}'] = df[f'bb_middle_{period}'] - (bb_std * std_dev)
                            df[f'bb_width_{period}_{std_dev}'] = (df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}']) / df[f'bb_middle_{period}']
                            df[f'bb_position_{period}_{std_dev}'] = (df['close'] - df[f'bb_lower_{period}_{std_dev}']) / (df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}'])
            
            # Stochastic (разные периоды)
            if any('stoch' in ind for ind in selected_indicators):
                for k_period, d_period in [(14, 3), (21, 5), (30, 10)]:
                    if f'stoch_{k_period}_{d_period}' in selected_indicators:
                        low_k = df['low'].rolling(window=k_period).min()
                        high_k = df['high'].rolling(window=k_period).max()
                        df[f'stoch_k_{k_period}'] = 100 * ((df['close'] - low_k) / (high_k - low_k))
                        df[f'stoch_d_{k_period}_{d_period}'] = df[f'stoch_k_{k_period}'].rolling(window=d_period).mean()
            
            # Williams %R
            if any('williams' in ind for ind in selected_indicators):
                for period in [14, 21, 30]:
                    if f'williams_{period}' in selected_indicators:
                        high_period = df['high'].rolling(window=period).max()
                        low_period = df['low'].rolling(window=period).min()
                        df[f'williams_{period}'] = -100 * ((high_period - df['close']) / (high_period - low_period))
            
            # CCI (Commodity Channel Index)
            if any('cci' in ind for ind in selected_indicators):
                for period in [20, 30, 50]:
                    if f'cci_{period}' in selected_indicators:
                        typical_price = (df['high'] + df['low'] + df['close']) / 3
                        sma_tp = typical_price.rolling(window=period).mean()
                        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
                        df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # ADX (Average Directional Index)
            if any('adx' in ind for ind in selected_indicators):
                for period in [14, 21, 30]:
                    if f'adx_{period}' in selected_indicators:
                        high_diff = df['high'].diff()
                        low_diff = df['low'].diff()
                        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
                        minus_dm = -low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
                        
                        atr = self._calculate_atr(df, period)
                        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
                        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
                        
                        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                        df[f'adx_{period}'] = dx.rolling(window=period).mean()
            
            # Volume indicators (расширенные)
            if any('volume' in ind for ind in selected_indicators):
                for period in [10, 20, 50]:
                    if f'volume_sma_{period}' in selected_indicators:
                        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
                
                # OBV (On Balance Volume)
                if 'obv' in selected_indicators:
                    df['obv'] = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, 
                                                         np.where(df['close'] < df['close'].shift(1), -1, 0))).cumsum()
                
                # Volume Price Trend
                if 'vpt' in selected_indicators:
                    df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
            
            # Price momentum (расширенные)
            if any('momentum' in ind for ind in selected_indicators):
                for period in [1, 3, 5, 10, 20, 50]:
                    if f'momentum_{period}' in selected_indicators:
                        df[f'momentum_{period}'] = df['close'].pct_change(period)
                        df[f'momentum_abs_{period}'] = df[f'momentum_{period}'].abs()
            
            # Volatility indicators (расширенные)
            if any('volatility' in ind for ind in selected_indicators):
                for period in [5, 10, 20, 30, 50]:
                    if f'volatility_{period}' in selected_indicators:
                        df[f'volatility_{period}'] = df['close'].rolling(window=period).std()
                        df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(window=period*2).mean()
                
                # ATR (Average True Range)
                for period in [14, 21, 30]:
                    if f'atr_{period}' in selected_indicators:
                        df[f'atr_{period}'] = self._calculate_atr(df, period)
                        df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
            
            # Price patterns (расширенные)
            if any('pattern' in ind for ind in selected_indicators):
                df['price_change'] = df['close'].pct_change()
                df['price_change_abs'] = df['price_change'].abs()
                df['high_low_ratio'] = df['high'] / df['low']
                df['close_open_ratio'] = df['close'] / df['open']
                df['body_size'] = abs(df['close'] - df['open']) / df['open']
                df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
                df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
            
            # Trend indicators
            if any('trend' in ind for ind in selected_indicators):
                # Parabolic SAR
                if 'psar' in selected_indicators:
                    df['psar'] = self._calculate_psar(df)
                
                # Ichimoku Cloud
                if 'ichimoku' in selected_indicators:
                    df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
                    df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
                    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
                    df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
                    df['chikou_span'] = df['close'].shift(-26)
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания технических индикаторов: {e}")
            return data
    
    def get_all_available_indicators(self) -> List[str]:
        """Получить список всех доступных индикаторов"""
        return [
            # Moving Averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
            
            # RSI
            'rsi_14', 'rsi_21', 'rsi_30',
            
            # MACD
            'macd_12_26_9', 'macd_5_35_5', 'macd_8_21_5',
            
            # Bollinger Bands
            'bb_20_2', 'bb_20_2.5', 'bb_20_3', 'bb_30_2', 'bb_50_2',
            
            # Stochastic
            'stoch_14_3', 'stoch_21_5', 'stoch_30_10',
            
            # Williams %R
            'williams_14', 'williams_21', 'williams_30',
            
            # CCI
            'cci_20', 'cci_30', 'cci_50',
            
            # ADX
            'adx_14', 'adx_21', 'adx_30',
            
            # Volume
            'volume_sma_10', 'volume_sma_20', 'volume_sma_50', 'obv', 'vpt',
            
            # Momentum
            'momentum_1', 'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
            
            # Volatility
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_30', 'volatility_50',
            'atr_14', 'atr_21', 'atr_30',
            
            # Patterns
            'pattern',
            
            # Trend
            'psar', 'ichimoku'
        ]
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Расчет Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_psar(self, df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, af_maximum: float = 0.2) -> pd.Series:
        """Расчет Parabolic SAR"""
        psar = pd.Series(index=df.index, dtype=float)
        af = af_start
        ep = df['low'].iloc[0]
        trend = 1
        
        for i in range(1, len(df)):
            if trend == 1:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if df['low'].iloc[i] < psar.iloc[i]:
                    trend = -1
                    psar.iloc[i] = ep
                    ep = df['high'].iloc[i]
                    af = af_start
                else:
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af = min(af + af_increment, af_maximum)
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if df['high'].iloc[i] > psar.iloc[i]:
                    trend = 1
                    psar.iloc[i] = ep
                    ep = df['low'].iloc[i]
                    af = af_start
                else:
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af = min(af + af_increment, af_maximum)
        
        return psar
    
    def optimize_indicators_for_data(self, data: pd.DataFrame, symbol: str = None) -> List[str]:
        """Оптимизация индикаторов для конкретных данных"""
        if not self.optimize_indicators or not self.indicator_optimizer:
            # Возвращаем базовые индикаторы если оптимизация отключена
            return ['sma_20', 'ema_20', 'rsi_14', 'macd_12_26_9', 'bb_20_2', 'stoch_14_3']
        
        try:
            logger.info(f"Оптимизация индикаторов для {symbol or 'данных'}...")
            
            # Оценка эффективности всех индикаторов
            scores = self.indicator_optimizer.evaluate_indicators(data)
            
            if not scores:
                logger.warning("Не удалось оценить индикаторы, используем базовые")
                return ['sma_20', 'ema_20', 'rsi_14', 'macd_12_26_9', 'bb_20_2', 'stoch_14_3']
            
            # Выбор лучших индикаторов
            best_indicators = self.indicator_optimizer.select_best_indicators(
                scores, 
                top_n=self.max_indicators,
                min_correlation=0.05
            )
            
            # Сохраняем результаты
            self.indicator_scores[symbol or 'default'] = scores
            self.optimized_indicators = best_indicators
            
            logger.info(f"Выбрано {len(best_indicators)} лучших индикаторов:")
            for i, indicator in enumerate(best_indicators[:10], 1):
                score = scores.get(indicator, 0)
                logger.info(f"  {i}. {indicator}: {score:.4f}")
            
            return best_indicators
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации индикаторов: {e}")
            return ['sma_20', 'ema_20', 'rsi_14', 'macd_12_26_9', 'bb_20_2', 'stoch_14_3']
    
    def get_optimized_indicators(self, symbol: str = None) -> List[str]:
        """Получить оптимизированные индикаторы"""
        if self.optimized_indicators:
            return self.optimized_indicators
        
        # Если нет оптимизированных, возвращаем базовые
        return ['sma_20', 'ema_20', 'rsi_14', 'macd_12_26_9', 'bb_20_2', 'stoch_14_3']
    
    async def _send_trade_notification(self, symbol: str, action: str, quantity: int, 
                                     price: float, strategy: str, pnl: float = None,
                                     reason: str = None, confidence: float = None):
        """Отправка уведомления о сделке"""
        if not self.enable_telegram or not self.telegram_notifier:
            return
        
        try:
            await self.telegram_notifier.notify_trade(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                strategy=strategy,
                pnl=pnl,
                reason=reason,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления о сделке: {e}")
    
    async def _send_portfolio_notification(self, total_value: float, total_pnl: float,
                                         positions_count: int, daily_pnl: float = 0.0):
        """Отправка уведомления об обновлении портфеля"""
        if not self.enable_telegram or not self.telegram_notifier:
            return
        
        try:
            await self.telegram_notifier.notify_portfolio_update(
                total_value=total_value,
                total_pnl=total_pnl,
                positions_count=positions_count,
                daily_pnl=daily_pnl
            )
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления о портфеле: {e}")
    
    async def _send_alert(self, alert_type: str, title: str, message: str, severity: str = 'INFO'):
        """Отправка алерта"""
        if not self.enable_telegram or not self.telegram_notifier:
            return
        
        try:
            await self.telegram_notifier.notify_alert(
                alert_type=alert_type,
                title=title,
                message=message,
                severity=severity
            )
        except Exception as e:
            logger.error(f"Ошибка отправки алерта: {e}")
    
    def arima_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """ARIMA стратегия для прогнозирования цен"""
        try:
            logger.info(f"Тестирование ARIMA стратегии для {symbol}")
            
            # Оптимизируем индикаторы для данных
            optimized_indicators = self.optimize_indicators_for_data(data, symbol)
            
            # Подготавливаем данные с оптимизированными индикаторами
            df = self.create_technical_features(data, optimized_indicators)
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
                                
                                # Отправка уведомления о покупке
                                if self.enable_telegram:
                                    asyncio.create_task(self._send_trade_notification(
                                        symbol=symbol,
                                        action="BUY",
                                        quantity=position_size,
                                        price=current_price,
                                        strategy="ARIMA",
                                        confidence=confidence
                                    ))
                        
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
                            
                            # Отправка уведомления о продаже
                            if self.enable_telegram:
                                asyncio.create_task(self._send_trade_notification(
                                    symbol=symbol,
                                    action="SELL",
                                    quantity=position['size'],
                                    price=exit_price,
                                    strategy="ARIMA",
                                    pnl=pnl,
                                    reason="SIGNAL"
                                ))
                
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
        
        # Получаем настройки из переменных окружения
        trading_symbols = os.getenv('TRADING_SYMBOLS', 'GAZP,SBER,PIKK,IRAO,SGZH').split(',')
        trading_period = os.getenv('TRADING_PERIOD', '1Y')
        min_data_days = int(os.getenv('MIN_DATA_DAYS', '100'))
        
        logger.info(f"Настройки торговли:")
        logger.info(f"  Инструменты: {trading_symbols}")
        logger.info(f"  Период: {trading_period}")
        logger.info(f"  Минимум дней: {min_data_days}")
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_tbank.csv'):
                symbol_period = filename.replace('_tbank.csv', '')
                parts = symbol_period.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    period = parts[1]
                    
                    # Проверяем, включен ли инструмент в список для торговли
                    if symbol not in trading_symbols:
                        logger.debug(f"Инструмент {symbol} не включен в список для торговли")
                        continue
                    
                    # Проверяем, соответствует ли период
                    if period != trading_period:
                        logger.debug(f"Период {period} не соответствует настройке {trading_period}")
                        continue
                    
                    filepath = os.path.join(data_dir, filename)
                    try:
                        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
                        if not df.empty and len(df) >= min_data_days:
                            key = f"{symbol}_{period}"
                            market_data[key] = df
                            logger.info(f"✅ Загружены данные {key}: {len(df)} дней")
                        else:
                            logger.warning(f"⚠️ Недостаточно данных для {symbol}_{period}: {len(df) if not df.empty else 0} дней (требуется {min_data_days})")
                    except Exception as e:
                        logger.error(f"Ошибка загрузки {filename}: {e}")
        
        logger.info(f"Итого загружено инструментов: {len(market_data)}")
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
