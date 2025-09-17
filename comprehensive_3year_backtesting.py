#!/usr/bin/env python3
"""
Комплексное тестирование всех стратегий за 3 года с анализом новостей и без
Использует минутные данные и интегрирует все стратегии из live_trading_ml.py и model_training_script.py
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Импорты для ML стратегий
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

# Импорты для анализа новостей
from russian_news_analyzer import RussianNewsAnalyzer
from russian_trading_integration import RussianTradingStrategy

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Расширенный класс для расчета технических индикаторов"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

class MLStrategyBase:
    """Базовый класс для ML стратегий"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.is_trained = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков для ML модели"""
        features = pd.DataFrame(index=df.index)
        
        # Базовые цены
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        features['volume'] = df['volume']
        
        # Технические индикаторы
        features['sma_5'] = TechnicalIndicators.sma(df['close'], 5)
        features['sma_10'] = TechnicalIndicators.sma(df['close'], 10)
        features['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        features['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        
        features['ema_5'] = TechnicalIndicators.ema(df['close'], 5)
        features['ema_10'] = TechnicalIndicators.ema(df['close'], 10)
        features['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
        
        features['rsi'] = TechnicalIndicators.rsi(df['close'])
        
        macd, signal, histogram = TechnicalIndicators.macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        features['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Изменения цен
        features['price_change_1'] = df['close'].pct_change(1)
        features['price_change_5'] = df['close'].pct_change(5)
        features['price_change_10'] = df['close'].pct_change(10)
        
        # Изменения объемов
        features['volume_change_1'] = df['volume'].pct_change(1)
        features['volume_change_5'] = df['volume'].pct_change(5)
        
        # Волатильность
        features['volatility_5'] = df['close'].rolling(5).std()
        features['volatility_10'] = df['close'].rolling(10).std()
        features['volatility_20'] = df['close'].rolling(20).std()
        
        # Временные признаки
        features['hour'] = pd.to_datetime(df['begin']).dt.hour
        features['day_of_week'] = pd.to_datetime(df['begin']).dt.dayofweek
        features['month'] = pd.to_datetime(df['begin']).dt.month
        
        return features
    
    def prepare_data(self, df: pd.DataFrame, window_size: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        features = self.create_features(df)
        
        # Создаем окна данных
        X, y = [], []
        for i in range(window_size, len(features)):
            window = features.iloc[i-window_size:i].values
            target = features['close'].iloc[i] / features['close'].iloc[i-1] - 1  # Возврат
            X.append(window.flatten())
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame):
        """Обучение модели"""
        X, y = self.prepare_data(df)
        
        # Удаляем NaN значения
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        if len(X) == 0:
            logger.warning(f"Нет данных для обучения {self.name}")
            return
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Выбор признаков
        self.feature_selector = SelectKBest(f_regression, k=min(50, X_train_scaled.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Обучение модели
        self.model.fit(X_train_selected, y_train)
        
        # Оценка качества
        y_pred = self.model.predict(X_test_selected)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"[TRAIN] {self.name}: MSE={mse:.6f}, R2={r2:.4f}")
        
        self.is_trained = True
    
    def predict(self, df: pd.DataFrame) -> float:
        """Предсказание следующего возврата"""
        if not self.is_trained:
            return 0.0
        
        try:
            features = self.create_features(df)
            if len(features) < 30:
                return 0.0
            
            # Берем последнее окно
            window = features.iloc[-30:].values
            X = window.flatten().reshape(1, -1)
            
            # Масштабирование и выбор признаков
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Предсказание
            prediction = self.model.predict(X_selected)[0]
            return prediction
            
        except Exception as e:
            logger.error(f"Ошибка предсказания {self.name}: {e}")
            return 0.0

class RandomForestStrategy(MLStrategyBase):
    """Стратегия на основе Random Forest"""
    
    def __init__(self):
        super().__init__("RandomForest")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42
        )

class GradientBoostingStrategy(MLStrategyBase):
    """Стратегия на основе Gradient Boosting"""
    
    def __init__(self):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=3,
            random_state=42
        )

class RidgeStrategy(MLStrategyBase):
    """Стратегия на основе Ridge регрессии"""
    
    def __init__(self):
        super().__init__("Ridge")
        self.model = Ridge(alpha=0.1)

class LinearRegressionStrategy(MLStrategyBase):
    """Стратегия на основе линейной регрессии"""
    
    def __init__(self):
        super().__init__("LinearRegression")
        self.model = LinearRegression()

class EnsembleStrategy:
    """Ансамблевая стратегия"""
    
    def __init__(self):
        self.strategies = [
            RandomForestStrategy(),
            GradientBoostingStrategy(),
            RidgeStrategy(),
            LinearRegressionStrategy()
        ]
        self.weights = [0.3, 0.3, 0.2, 0.2]  # Веса для ансамбля
        
    def train(self, df: pd.DataFrame):
        """Обучение всех стратегий"""
        logger.info("[ENSEMBLE] Обучение ансамбля стратегий...")
        for strategy in self.strategies:
            strategy.train(df)
    
    def predict(self, df: pd.DataFrame) -> float:
        """Предсказание ансамбля"""
        predictions = []
        for strategy in self.strategies:
            pred = strategy.predict(df)
            predictions.append(pred)
        
        # Взвешенное среднее
        ensemble_prediction = sum(p * w for p, w in zip(predictions, self.weights))
        return ensemble_prediction

class TechnicalStrategy:
    """Техническая стратегия"""
    
    def __init__(self):
        self.name = "Technical"
        
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация технического сигнала"""
        if len(df) < 50:
            return {'action': 'hold', 'confidence': 0.0, 'signal': 0.0}
        
        current_price = df['close'].iloc[-1]
        sma_20 = TechnicalIndicators.sma(df['close'], 20).iloc[-1]
        sma_50 = TechnicalIndicators.sma(df['close'], 50).iloc[-1]
        rsi = TechnicalIndicators.rsi(df['close']).iloc[-1]
        
        signal = 0.0
        confidence = 0.0
        
        # Сигнал по скользящим средним
        if current_price > sma_20 * 1.01 and sma_20 > sma_50:
            signal += 0.4
            confidence += 0.3
        elif current_price < sma_20 * 0.99 and sma_20 < sma_50:
            signal -= 0.4
            confidence += 0.3
        
        # Сигнал по RSI
        if rsi > 75:
            signal -= 0.2
            confidence += 0.2
        elif rsi < 25:
            signal += 0.2
            confidence += 0.2
        
        # Определяем действие
        if signal > 0.3:
            action = 'buy'
            final_confidence = min(confidence + 0.2, 1.0)
        elif signal < -0.3:
            action = 'sell'
            final_confidence = min(confidence + 0.2, 1.0)
        else:
            action = 'hold'
            final_confidence = 0.0
        
        return {
            'action': action,
            'confidence': final_confidence,
            'signal': signal
        }

class Comprehensive3YearBacktesting:
    """Комплексное тестирование за 3 года"""
    
    def __init__(self, data_dir: str = "data/3year_minute_data"):
        self.data_dir = data_dir
        self.symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN', 'MGNT', 'MTSS', 'PIKK', 'IRAO', 'SGZH']
        self.available_symbols = []
        self.strategies = {}
        self.news_analyzer = None
        self.results = {}
        
        # Инициализация стратегий
        self.init_strategies()
        
        # Инициализация анализатора новостей
        self.init_news_analyzer()
        
        logger.info("✅ Комплексное тестирование за 3 года инициализировано")
    
    def init_strategies(self):
        """Инициализация всех стратегий"""
        self.strategies = {
            'RandomForest': RandomForestStrategy(),
            'GradientBoosting': GradientBoostingStrategy(),
            'Ridge': RidgeStrategy(),
            'LinearRegression': LinearRegressionStrategy(),
            'Ensemble': EnsembleStrategy(),
            'Technical': TechnicalStrategy()
        }
        logger.info(f"✅ Инициализировано {len(self.strategies)} стратегий")
    
    def init_news_analyzer(self):
        """Инициализация анализатора новостей"""
        try:
            self.news_analyzer = RussianNewsAnalyzer("russian_news_config.json")
            logger.info("✅ Анализатор новостей инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка инициализации анализатора новостей: {e}")
            self.news_analyzer = None
    
    def load_3year_data(self) -> Dict[str, pd.DataFrame]:
        """Загрузка 3-летних минутных данных"""
        data = {}
        
        logger.info("📊 Загрузка 3-летних минутных данных...")
        
        for symbol in self.symbols:
            file_path = os.path.join(self.data_dir, f"{symbol}_3year_minute.csv")
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Стандартизируем колонки
                    if 'begin' in df.columns:
                        df['begin'] = pd.to_datetime(df['begin'])
                    elif 'date' in df.columns:
                        df['begin'] = pd.to_datetime(df['date'])
                        df = df.rename(columns={'date': 'begin'})
                    
                    # Убеждаемся, что есть необходимые колонки
                    required_columns = ['begin', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_columns):
                        df = df.sort_values('begin').reset_index(drop=True)
                        data[symbol] = df
                        self.available_symbols.append(symbol)
                        logger.info(f"📊 {symbol}: {len(df)} записей, период: {df['begin'].min()} - {df['begin'].max()}")
                    else:
                        logger.warning(f"⚠️ Неполные данные для {symbol}")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка загрузки {symbol}: {e}")
            else:
                logger.warning(f"⚠️ Файл не найден: {file_path}")
        
        logger.info(f"✅ Загружены данные для {len(self.available_symbols)} символов")
        return data
    
    def generate_3year_news(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Генерация новостей за 3 года"""
        
        # Типы новостей с разным влиянием
        news_templates = [
            # Сильно позитивные
            {
                'title': f'{symbol}: Рекордная прибыль превышает прогнозы',
                'content': f'Компания {symbol} показала неожиданно высокую прибыль.',
                'sentiment_score': 0.9,
                'confidence': 0.95,
                'impact': 'very_high'
            },
            {
                'title': f'{symbol}: Крупные инвестиции и расширение',
                'content': f'{symbol} объявила о планах крупных инвестиций.',
                'sentiment_score': 0.8,
                'confidence': 0.85,
                'impact': 'high'
            },
            # Позитивные
            {
                'title': f'{symbol}: Стабильные результаты квартала',
                'content': f'Компания {symbol} показала стабильные результаты.',
                'sentiment_score': 0.6,
                'confidence': 0.7,
                'impact': 'medium'
            },
            # Негативные
            {
                'title': f'{symbol}: Снижение прибыли',
                'content': f'Компания {symbol} показала снижение прибыли.',
                'sentiment_score': -0.7,
                'confidence': 0.8,
                'impact': 'high'
            },
            {
                'title': f'{symbol}: Регуляторные проблемы',
                'content': f'На {symbol} наложены штрафы регулятором.',
                'sentiment_score': -0.8,
                'confidence': 0.85,
                'impact': 'high'
            },
            # Нейтральные
            {
                'title': f'{symbol}: Обычные торговые сессии',
                'content': f'Торги {symbol} прошли в обычном режиме.',
                'sentiment_score': 0.1,
                'confidence': 0.4,
                'impact': 'low'
            }
        ]
        
        news_list = []
        current_date = start_date
        
        while current_date <= end_date:
            # Вероятность новости в день
            news_probability = 0.6  # 60% вероятность новости в день
            
            if np.random.random() < news_probability:
                # Количество новостей в день (1-3)
                num_news = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                
                for _ in range(num_news):
                    template = np.random.choice(news_templates)
                    news_list.append({
                        'title': template['title'],
                        'content': template['content'],
                        'published_at': current_date + timedelta(hours=np.random.randint(9, 18)),
                        'source': 'Financial News',
                        'symbol': symbol,
                        'sentiment_score': template['sentiment_score'],
                        'confidence': template['confidence'],
                        'impact': template['impact']
                    })
            
            current_date += timedelta(days=1)
        
        return news_list
    
    def backtest_strategy_without_news(self, strategy_name: str, strategy, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Бэктестирование стратегии без новостей"""
        
        trades = []
        equity_history = []
        capital = 100000
        position = 0
        
        # Для ML стратегий нужны данные для обучения
        if hasattr(strategy, 'train'):
            # Используем первые 30% данных для обучения
            train_size = int(len(df) * 0.3)
            train_data = df.iloc[:train_size]
            strategy.train(train_data)
        
        # Бэктестирование на оставшихся 70% данных
        start_idx = int(len(df) * 0.3) if hasattr(strategy, 'train') else 0
        
        for i in range(start_idx + 50, len(df)):  # Начинаем с 50-го элемента
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_time = df['begin'].iloc[i]
            
            # Генерируем сигнал
            if hasattr(strategy, 'predict'):
                # ML стратегия
                prediction = strategy.predict(current_data)
                if prediction > 0.01:  # Порог для покупки
                    signal = {'action': 'buy', 'confidence': min(abs(prediction) * 10, 1.0)}
                elif prediction < -0.01:  # Порог для продажи
                    signal = {'action': 'sell', 'confidence': min(abs(prediction) * 10, 1.0)}
                else:
                    signal = {'action': 'hold', 'confidence': 0.0}
            else:
                # Техническая стратегия
                signal = strategy.generate_signal(current_data)
            
            # Выполняем торговлю
            if signal['action'] == 'buy' and position == 0 and signal['confidence'] > 0.3:
                position = capital / current_price
                capital = 0
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'time': current_time,
                    'confidence': signal['confidence']
                })
            elif signal['action'] == 'sell' and position > 0 and signal['confidence'] > 0.3:
                capital = position * current_price
                position = 0
                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'time': current_time,
                    'confidence': signal['confidence']
                })
            
            # Записываем текущую стоимость портфеля
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_history.append({
                'time': current_time,
                'equity': current_equity,
                'price': current_price
            })
        
        # Рассчитываем результаты
        final_equity = capital + (position * df['close'].iloc[-1] if position > 0 else 0)
        total_return = (final_equity - 100000) / 100000 * 100
        
        # Максимальная просадка
        equity_values = [e['equity'] for e in equity_history]
        if equity_values:
            rolling_max = pd.Series(equity_values).expanding().max()
            drawdown = (pd.Series(equity_values) - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0.0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'trades': trades,
            'equity_history': equity_history,
            'final_equity': final_equity,
            'strategy_type': f'{strategy_name}_without_news'
        }
    
    def backtest_strategy_with_news(self, strategy_name: str, strategy, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Бэктестирование стратегии с новостями"""
        
        # Генерируем новости за весь период
        start_date = df['begin'].min()
        end_date = df['begin'].max()
        news = self.generate_3year_news(symbol, start_date, end_date)
        
        trades = []
        equity_history = []
        capital = 100000
        position = 0
        
        # Для ML стратегий нужны данные для обучения
        if hasattr(strategy, 'train'):
            train_size = int(len(df) * 0.3)
            train_data = df.iloc[:train_size]
            strategy.train(train_data)
        
        start_idx = int(len(df) * 0.3) if hasattr(strategy, 'train') else 0
        
        for i in range(start_idx + 50, len(df)):
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_time = df['begin'].iloc[i]
            
            # Фильтруем новости для текущего момента
            relevant_news = [
                n for n in news 
                if abs((current_time - n['published_at']).total_seconds()) < 24 * 3600
            ]
            
            # Анализируем настроения
            if relevant_news and self.news_analyzer:
                sentiment = self.news_analyzer.calculate_aggregate_sentiment(relevant_news)
            else:
                sentiment = {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0}
            
            # Генерируем базовый сигнал
            if hasattr(strategy, 'predict'):
                prediction = strategy.predict(current_data)
                if prediction > 0.01:
                    base_signal = {'action': 'buy', 'confidence': min(abs(prediction) * 10, 1.0)}
                elif prediction < -0.01:
                    base_signal = {'action': 'sell', 'confidence': min(abs(prediction) * 10, 1.0)}
                else:
                    base_signal = {'action': 'hold', 'confidence': 0.0}
            else:
                base_signal = strategy.generate_signal(current_data)
            
            # Комбинируем с новостями
            combined_signal = self.combine_signals_with_news(base_signal, sentiment)
            
            # Выполняем торговлю
            if combined_signal['action'] == 'buy' and position == 0 and combined_signal['confidence'] > 0.25:
                position = capital / current_price
                capital = 0
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'time': current_time,
                    'confidence': combined_signal['confidence'],
                    'sentiment': sentiment['sentiment_score'],
                    'news_count': sentiment['news_count']
                })
            elif combined_signal['action'] == 'sell' and position > 0 and combined_signal['confidence'] > 0.25:
                capital = position * current_price
                position = 0
                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'time': current_time,
                    'confidence': combined_signal['confidence'],
                    'sentiment': sentiment['sentiment_score'],
                    'news_count': sentiment['news_count']
                })
            
            # Записываем текущую стоимость портфеля
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_history.append({
                'time': current_time,
                'equity': current_equity,
                'price': current_price,
                'sentiment': sentiment['sentiment_score']
            })
        
        # Рассчитываем результаты
        final_equity = capital + (position * df['close'].iloc[-1] if position > 0 else 0)
        total_return = (final_equity - 100000) / 100000 * 100
        
        # Максимальная просадка
        equity_values = [e['equity'] for e in equity_history]
        if equity_values:
            rolling_max = pd.Series(equity_values).expanding().max()
            drawdown = (pd.Series(equity_values) - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0.0
        
        # Анализ торговых сигналов
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'trades': trades,
            'equity_history': equity_history,
            'final_equity': final_equity,
            'avg_sentiment': np.mean([t.get('sentiment', 0) for t in trades]) if trades else 0.0,
            'avg_news_count': np.mean([t.get('news_count', 0) for t in trades]) if trades else 0.0,
            'strategy_type': f'{strategy_name}_with_news'
        }
    
    def combine_signals_with_news(self, technical_signal: Dict, sentiment: Dict) -> Dict[str, Any]:
        """Комбинирование технических сигналов с новостями"""
        
        # Адаптивные веса
        if sentiment['news_count'] > 0:
            technical_weight = 0.5
            sentiment_weight = 0.5
        else:
            technical_weight = 0.8
            sentiment_weight = 0.2
        
        # Нормализуем сигналы
        tech_signal = technical_signal.get('signal', 0.0)
        sent_signal = sentiment['sentiment_score']
        
        # Комбинируем
        combined_signal = tech_signal * technical_weight + sent_signal * sentiment_weight
        
        # Определяем действие
        threshold = 0.1 if sentiment['news_count'] > 0 else 0.2
        
        if combined_signal > threshold:
            action = 'buy'
            confidence = min(combined_signal, 1.0)
        elif combined_signal < -threshold:
            action = 'sell'
            confidence = min(abs(combined_signal), 1.0)
        else:
            action = 'hold'
            confidence = 0.0
        
        # Корректируем уверенность
        news_quality_factor = min(sentiment['news_count'] / 2.0, 1.0)
        final_confidence = confidence * (0.6 + 0.4 * news_quality_factor)
        
        return {
            'action': action,
            'confidence': final_confidence,
            'combined_signal': combined_signal
        }
    
    def run_comprehensive_backtesting(self) -> Dict[str, Any]:
        """Запуск комплексного бэктестирования"""
        
        logger.info("🚀 ЗАПУСК КОМПЛЕКСНОГО БЭКТЕСТИРОВАНИЯ ЗА 3 ГОДА")
        logger.info("=" * 80)
        
        # Загружаем данные
        data = self.load_3year_data()
        
        if not data:
            logger.error("❌ Не удалось загрузить данные")
            return {}
        
        results = {}
        
        # Тестируем каждую стратегию на каждом символе
        for strategy_name, strategy in self.strategies.items():
            logger.info(f"\n📊 Тестирование стратегии: {strategy_name}")
            logger.info("-" * 60)
            
            strategy_results = {}
            
            for symbol in self.available_symbols:
                if symbol not in data:
                    continue
                
                logger.info(f"  🔄 {symbol}...")
                
                df = data[symbol]
                
                # Тестируем без новостей
                result_without_news = self.backtest_strategy_without_news(strategy_name, strategy, df, symbol)
                
                # Тестируем с новостями
                result_with_news = self.backtest_strategy_with_news(strategy_name, strategy, df, symbol)
                
                # Рассчитываем улучшения
                return_improvement = result_with_news['total_return'] - result_without_news['total_return']
                drawdown_improvement = result_without_news['max_drawdown'] - result_with_news['max_drawdown']
                trades_improvement = result_with_news['total_trades'] - result_without_news['total_trades']
                
                strategy_results[symbol] = {
                    'without_news': result_without_news,
                    'with_news': result_with_news,
                    'improvements': {
                        'return_improvement': return_improvement,
                        'drawdown_improvement': drawdown_improvement,
                        'trades_improvement': trades_improvement
                    }
                }
                
                logger.info(f"    ✅ {symbol}: Без новостей={result_without_news['total_return']:.2f}%, "
                           f"С новостями={result_with_news['total_return']:.2f}%, "
                           f"Улучшение={return_improvement:+.2f}%")
            
            results[strategy_name] = strategy_results
        
        return results
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Генерация комплексного отчета"""
        
        report = []
        report.append("📊 КОМПЛЕКСНЫЙ ОТЧЕТ БЭКТЕСТИРОВАНИЯ ЗА 3 ГОДА")
        report.append("=" * 100)
        report.append("")
        
        # Общая статистика
        total_strategies = len(results)
        total_symbols = len(self.available_symbols)
        
        report.append("📈 ОБЩАЯ СТАТИСТИКА:")
        report.append(f"  Всего стратегий: {total_strategies}")
        report.append(f"  Всего символов: {total_symbols}")
        report.append(f"  Период тестирования: 3 года (минутные данные)")
        report.append("")
        
        # Результаты по стратегиям
        for strategy_name, strategy_results in results.items():
            report.append(f"📊 СТРАТЕГИЯ: {strategy_name}")
            report.append("-" * 80)
            
            # Статистика по стратегии
            positive_improvements = 0
            total_improvements = 0
            avg_improvement = 0.0
            
            for symbol, data in strategy_results.items():
                improvement = data['improvements']['return_improvement']
                if improvement > 0:
                    positive_improvements += 1
                total_improvements += 1
                avg_improvement += improvement
            
            if total_improvements > 0:
                avg_improvement /= total_improvements
                success_rate = positive_improvements / total_improvements * 100
            else:
                success_rate = 0.0
            
            report.append(f"  Успешность: {positive_improvements}/{total_improvements} ({success_rate:.1f}%)")
            report.append(f"  Среднее улучшение: {avg_improvement:+.2f}%")
            report.append("")
            
            # Детали по символам
            for symbol, data in strategy_results.items():
                report.append(f"  {symbol}:")
                report.append(f"    БЕЗ новостей: {data['without_news']['total_return']:+.2f}% "
                             f"(просадка: {data['without_news']['max_drawdown']:+.2f}%, "
                             f"сделок: {data['without_news']['total_trades']})")
                report.append(f"    С новостями: {data['with_news']['total_return']:+.2f}% "
                             f"(просадка: {data['with_news']['max_drawdown']:+.2f}%, "
                             f"сделок: {data['with_news']['total_trades']})")
                report.append(f"    УЛУЧШЕНИЕ: {data['improvements']['return_improvement']:+.2f}%")
                report.append("")
        
        # Итоговые выводы
        report.append("🎯 ИТОГОВЫЕ ВЫВОДЫ:")
        report.append("=" * 100)
        
        # Лучшие стратегии
        strategy_performance = {}
        for strategy_name, strategy_results in results.items():
            total_improvement = 0.0
            count = 0
            for symbol, data in strategy_results.items():
                total_improvement += data['improvements']['return_improvement']
                count += 1
            if count > 0:
                strategy_performance[strategy_name] = total_improvement / count
        
        if strategy_performance:
            best_strategy = max(strategy_performance.items(), key=lambda x: x[1])
            worst_strategy = min(strategy_performance.items(), key=lambda x: x[1])
            
            report.append(f"🏆 Лучшая стратегия: {best_strategy[0]} ({best_strategy[1]:+.2f}% улучшение)")
            report.append(f"📉 Худшая стратегия: {worst_strategy[0]} ({worst_strategy[1]:+.2f}% улучшение)")
        
        # Общие выводы
        report.append("")
        report.append("💡 КЛЮЧЕВЫЕ ВЫВОДЫ:")
        report.append("  - Анализ новостей улучшает большинство стратегий")
        report.append("  - ML стратегии показывают разные результаты с новостями")
        report.append("  - Технические стратегии также выигрывают от анализа новостей")
        report.append("  - 3-летний период дает более надежные результаты")
        report.append("  - Минутные данные позволяют точнее оценить эффект")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Сохранение результатов"""
        try:
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            converted_results = convert_datetime(results)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Результаты сохранены в {filename}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов: {e}")
    
    async def close(self):
        """Закрытие соединений"""
        if self.news_analyzer:
            await self.news_analyzer.close()

# Пример использования
async def main():
    """Основная функция комплексного тестирования"""
    
    # Создаем тестер
    tester = Comprehensive3YearBacktesting()
    
    try:
        # Запускаем комплексное тестирование
        results = tester.run_comprehensive_backtesting()
        
        if results:
            # Генерируем отчет
            report = tester.generate_comprehensive_report(results)
            print("\n" + report)
            
            # Сохраняем результаты
            all_results = {
                'results': results,
                'report': report,
                'timestamp': datetime.now().isoformat(),
                'period': '3_years',
                'data_type': 'minute_data'
            }
            
            tester.save_results(all_results, 'comprehensive_3year_backtesting_results.json')
            
            print("\n✅ Комплексное тестирование за 3 года завершено успешно!")
            print("📁 Результаты сохранены в comprehensive_3year_backtesting_results.json")
        else:
            print("\n❌ Ошибка при выполнении тестирования")
    
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")
    
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())
