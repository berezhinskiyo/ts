#!/usr/bin/env python3
"""
Продвинутые торговые роботы на базе TensorTrade с анализом новостей, CNN и различными подходами к обучению
Основано на: https://github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot
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
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# TensorTrade импорты
try:
    import tensortrade as tt
    from tensortrade.environments import TradingEnvironment
    from tensortrade.data import DataFeed, Stream
    from tensortrade.actions import DiscreteActionStrategy
    from tensortrade.rewards import SimpleProfitStrategy
    from tensortrade.exchanges import Exchange, ExchangeOptions
    from tensortrade.instruments import USD, BTC, ETH
    from tensortrade.wallets import Wallet, Portfolio
    TENSORTRADE_AVAILABLE = True
except ImportError:
    TENSORTRADE_AVAILABLE = False
    print("⚠️ TensorTrade не установлен. Установите: pip install tensortrade")

# ML импорты
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow не установлен. Установите: pip install tensorflow")

# NLP импорты для анализа новостей
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers не установлен. Установите: pip install transformers torch")

# Дополнительные импорты
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_trading_robots.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """Анализатор настроений новостей"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.financial_pipeline = None
        self.news_cache = {}
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Общая модель анализа настроений
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                
                # Специализированная модель для финансовых новостей
                self.financial_pipeline = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    return_all_scores=True
                )
                
                logger.info("✅ Модели анализа настроений загружены")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки моделей анализа настроений: {e}")
    
    def analyze_sentiment(self, text: str, use_financial: bool = True) -> Dict[str, float]:
        """Анализ настроения текста"""
        if not text or not self.sentiment_pipeline:
            return {'positive': 0.5, 'negative': 0.5, 'neutral': 0.0}
        
        try:
            # Используем финансовую модель если доступна
            pipeline_to_use = self.financial_pipeline if use_financial and self.financial_pipeline else self.sentiment_pipeline
            
            results = pipeline_to_use(text[:512])  # Ограничиваем длину
            
            # Преобразуем результаты в стандартный формат
            sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label or 'bullish' in label:
                    sentiment_scores['positive'] = score
                elif 'negative' in label or 'bearish' in label:
                    sentiment_scores['negative'] = score
                else:
                    sentiment_scores['neutral'] = score
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Ошибка анализа настроения: {e}")
            return {'positive': 0.5, 'negative': 0.5, 'neutral': 0.0}
    
    def get_market_news(self, symbols: List[str], days_back: int = 7) -> Dict[str, List[Dict]]:
        """Получение новостей по символам (заглушка для демонстрации)"""
        # В реальном проекте здесь был бы API для получения новостей
        # Например, NewsAPI, Alpha Vantage News, или Finam API
        
        news_data = {}
        
        for symbol in symbols:
            # Генерируем примеры новостей для демонстрации
            sample_news = [
                {
                    'title': f'{symbol} показывает рост на фоне позитивных отчетов',
                    'content': f'Акции {symbol} демонстрируют устойчивый рост благодаря улучшению финансовых показателей...',
                    'published_at': datetime.now() - timedelta(hours=np.random.randint(1, 24*days_back)),
                    'source': 'Financial News'
                },
                {
                    'title': f'Аналитики повышают прогнозы по {symbol}',
                    'content': f'Ведущие аналитики пересматривают свои прогнозы по {symbol} в сторону повышения...',
                    'published_at': datetime.now() - timedelta(hours=np.random.randint(1, 24*days_back)),
                    'source': 'Market Analysis'
                }
            ]
            
            news_data[symbol] = sample_news
        
        return news_data
    
    def calculate_sentiment_score(self, news_list: List[Dict]) -> float:
        """Расчет общего индекса настроений"""
        if not news_list:
            return 0.0
        
        total_sentiment = 0.0
        weight_sum = 0.0
        
        for news in news_list:
            # Анализируем заголовок и содержание
            title_sentiment = self.analyze_sentiment(news['title'])
            content_sentiment = self.analyze_sentiment(news['content'])
            
            # Взвешиваем (заголовок важнее)
            title_weight = 0.7
            content_weight = 0.3
            
            # Рассчитываем общий сентимент
            sentiment = (title_sentiment['positive'] - title_sentiment['negative']) * title_weight + \
                       (content_sentiment['positive'] - content_sentiment['negative']) * content_weight
            
            # Взвешиваем по времени (более свежие новости важнее)
            hours_ago = (datetime.now() - news['published_at']).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - hours_ago / (24 * 7))  # За неделю вес падает до 0.1
            
            total_sentiment += sentiment * time_weight
            weight_sum += time_weight
        
        return total_sentiment / weight_sum if weight_sum > 0 else 0.0

class CNNPatternClassifier:
    """Классификатор паттернов с помощью CNN"""
    
    def __init__(self, input_shape: Tuple[int, int] = (60, 5)):
        self.input_shape = input_shape
        self.model = None
        self.scaler = StandardScaler()
        self.pattern_types = ['bullish_flag', 'bearish_flag', 'head_shoulders', 'double_top', 'double_bottom', 'triangle', 'channel']
        
    def create_cnn_model(self) -> Model:
        """Создание CNN модели для классификации паттернов"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        # Входной слой
        input_layer = Input(shape=self.input_shape, name='price_data')
        
        # CNN слои для извлечения паттернов
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        
        conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        
        conv3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling1D(pool_size=2)(conv3)
        
        # Flatten и Dense слои
        flatten = Flatten()(pool3)
        dense1 = Dense(256, activation='relu')(flatten)
        dropout1 = Dropout(0.5)(dense1)
        dense2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        
        # Выходной слой для классификации паттернов
        pattern_output = Dense(len(self.pattern_types), activation='softmax', name='pattern_classification')(dropout2)
        
        # Дополнительный выход для предсказания направления цены
        direction_output = Dense(3, activation='softmax', name='direction_prediction')(dropout2)  # up, down, sideways
        
        # Создаем модель с двумя выходами
        model = Model(inputs=input_layer, outputs=[pattern_output, direction_output])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'pattern_classification': 'categorical_crossentropy',
                'direction_prediction': 'categorical_crossentropy'
            },
            loss_weights={'pattern_classification': 0.7, 'direction_prediction': 0.3},
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_training_data(self, df: pd.DataFrame, window_size: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Подготовка данных для обучения CNN"""
        # Нормализация данных
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        price_data = df[price_columns].values
        normalized_data = self.scaler.fit_transform(price_data)
        
        # Создание окон
        X = []
        pattern_labels = []
        direction_labels = []
        
        for i in range(window_size, len(normalized_data) - 1):
            # Окно цен
            window = normalized_data[i-window_size:i]
            X.append(window)
            
            # Генерируем метки паттернов (упрощенная версия)
            pattern_label = self._generate_pattern_label(df.iloc[i-window_size:i])
            pattern_labels.append(pattern_label)
            
            # Генерируем метки направления
            direction_label = self._generate_direction_label(df.iloc[i], df.iloc[i+1])
            direction_labels.append(direction_label)
        
        return np.array(X), np.array(pattern_labels), np.array(direction_labels)
    
    def _generate_pattern_label(self, window_df: pd.DataFrame) -> np.ndarray:
        """Генерация меток паттернов (упрощенная версия)"""
        # В реальном проекте здесь была бы сложная логика определения паттернов
        # Пока используем случайные метки для демонстрации
        
        label = np.zeros(len(self.pattern_types))
        pattern_idx = np.random.randint(0, len(self.pattern_types))
        label[pattern_idx] = 1.0
        
        return label
    
    def _generate_direction_label(self, current_row: pd.Series, next_row: pd.Series) -> np.ndarray:
        """Генерация меток направления движения цены"""
        price_change = (next_row['close'] - current_row['close']) / current_row['close']
        
        label = np.zeros(3)  # [up, down, sideways]
        
        if price_change > 0.01:  # Рост > 1%
            label[0] = 1.0  # up
        elif price_change < -0.01:  # Падение > 1%
            label[1] = 1.0  # down
        else:
            label[2] = 1.0  # sideways
        
        return label
    
    def train_model(self, X: np.ndarray, pattern_labels: np.ndarray, direction_labels: np.ndarray, 
                   epochs: int = 100, validation_split: float = 0.2) -> Dict:
        """Обучение CNN модели"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        self.model = self.create_cnn_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
        ]
        
        # Обучение
        history = self.model.fit(
            X,
            {'pattern_classification': pattern_labels, 'direction_prediction': direction_labels},
            epochs=epochs,
            validation_split=validation_split,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict_patterns(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Предсказание паттернов и направления"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        predictions = self.model.predict(X)
        pattern_predictions = predictions[0]
        direction_predictions = predictions[1]
        
        return pattern_predictions, direction_predictions

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
        # Технические индикаторы
        df = df.copy()
        
        # Простые скользящие средние
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price changes
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_20'] = df['close'].pct_change(20)
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std()
        
        # Выбираем признаки
        feature_columns = [
            'sma_5', 'sma_20', 'sma_50', 'rsi', 'bb_width',
            'volume_ratio', 'price_change_1', 'price_change_5', 
            'price_change_20', 'volatility'
        ]
        
        features = df[feature_columns].dropna()
        return features.values
    
    def train_unsupervised_models(self, df: pd.DataFrame):
        """Обучение моделей без учителя"""
        logger.info("🤖 Обучение моделей без учителя...")
        
        # Подготавливаем признаки
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
            raise ValueError("Модели не обучены")
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        anomalies = self.anomaly_detector.predict(features_pca)
        return anomalies
    
    def get_market_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Определение рыночных режимов"""
        if not self.is_trained:
            raise ValueError("Модели не обучены")
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        clusters = self.clusterer.predict(features_pca)
        return clusters

class AdvancedTradingRobot:
    """Продвинутый торговый робот с интеграцией всех компонентов"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 100000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        
        # Инициализация компонентов
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.pattern_classifier = CNNPatternClassifier()
        self.unsupervised_agent = UnsupervisedLearningAgent()
        
        # Параметры стратегии
        self.sentiment_weight = 0.3
        self.pattern_weight = 0.4
        self.technical_weight = 0.3
        
        logger.info("🤖 Продвинутый торговый робот инициализирован")
    
    def load_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Загрузка исторических данных"""
        logger.info(f"📥 Загружаем данные для {symbol}...")
        
        # Используем MOEX API
        base_url = "https://iss.moex.com/iss/"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_data = []
        current_date = start_date
        batch_size = 90
        
        while current_date < end_date:
            batch_end = min(current_date + timedelta(days=batch_size), end_date)
            
            try:
                url = f"{base_url}engines/stock/markets/shares/boards/TQBR/securities/{symbol}/candles.json"
                params = {
                    'from': current_date.strftime('%Y-%m-%d'),
                    'till': batch_end.strftime('%Y-%m-%d'),
                    'interval': 1
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
                        all_data.append(df_batch)
                
                current_date = batch_end
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Ошибка загрузки данных {symbol}: {e}")
                current_date = batch_end
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('begin').reset_index(drop=True)
            logger.info(f"✅ {symbol}: Загружено {len(df)} записей")
            return df
        else:
            logger.error(f"❌ Не удалось загрузить данные для {symbol}")
            return pd.DataFrame()
    
    def train_all_models(self, training_data: Dict[str, pd.DataFrame]):
        """Обучение всех моделей"""
        logger.info("🎓 Начинаем обучение всех моделей...")
        
        # Объединяем данные для обучения без учителя
        all_data = []
        for symbol, df in training_data.items():
            all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Обучение моделей без учителя
            self.unsupervised_agent.train_unsupervised_models(combined_df)
        
        # Обучение CNN для каждого символа
        for symbol, df in training_data.items():
            if len(df) > 200:  # Минимум данных
                logger.info(f"🎯 Обучение CNN для {symbol}...")
                
                try:
                    X, pattern_labels, direction_labels = self.pattern_classifier.prepare_training_data(df)
                    
                    if len(X) > 50:
                        history = self.pattern_classifier.train_model(X, pattern_labels, direction_labels, epochs=50)
                        logger.info(f"✅ CNN для {symbol} обучена")
                    else:
                        logger.warning(f"⚠️ Недостаточно данных для CNN {symbol}")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка обучения CNN для {symbol}: {e}")
    
    def generate_trading_signal(self, symbol: str, df: pd.DataFrame, 
                              sentiment_score: float) -> Dict[str, Any]:
        """Генерация торгового сигнала"""
        try:
            # 1. Анализ паттернов CNN
            pattern_signal = 0.0
            if self.pattern_classifier.model is not None:
                # Подготавливаем последние данные
                recent_data = df.tail(60)
                if len(recent_data) >= 60:
                    X = self.pattern_classifier.prepare_training_data(recent_data)[0]
                    if len(X) > 0:
                        pattern_pred, direction_pred = self.pattern_classifier.predict_patterns(X[-1:])
                        
                        # Интерпретируем предсказание направления
                        direction_probs = direction_pred[0]
                        pattern_signal = direction_probs[0] - direction_probs[1]  # up - down
            
            # 2. Анализ аномалий и режимов
            anomaly_signal = 0.0
            regime_signal = 0.0
            
            if self.unsupervised_agent.is_trained:
                anomalies = self.unsupervised_agent.detect_anomalies(df.tail(100))
                regimes = self.unsupervised_agent.get_market_regimes(df.tail(100))
                
                # Аномалии могут указывать на разворот
                if len(anomalies) > 0 and anomalies[-1] == -1:
                    anomaly_signal = 0.5  # Потенциальный разворот
                
                # Анализ режимов (упрощенный)
                if len(regimes) > 0:
                    current_regime = regimes[-1]
                    # Разные режимы могут иметь разные сигналы
                    regime_signal = (current_regime - 2) * 0.2  # Центрируем вокруг 0
            
            # 3. Технические индикаторы
            technical_signal = self._calculate_technical_signals(df)
            
            # 4. Объединяем все сигналы
            final_signal = (
                sentiment_score * self.sentiment_weight +
                pattern_signal * self.pattern_weight +
                technical_signal * self.technical_weight +
                anomaly_signal * 0.1 +
                regime_signal * 0.1
            )
            
            # Определяем действие
            if final_signal > 0.3:
                action = 'buy'
                confidence = min(final_signal, 1.0)
            elif final_signal < -0.3:
                action = 'sell'
                confidence = min(abs(final_signal), 1.0)
            else:
                action = 'hold'
                confidence = 0.0
            
            return {
                'action': action,
                'confidence': confidence,
                'final_signal': final_signal,
                'components': {
                    'sentiment': sentiment_score,
                    'pattern': pattern_signal,
                    'technical': technical_signal,
                    'anomaly': anomaly_signal,
                    'regime': regime_signal
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации сигнала для {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'final_signal': 0.0}
    
    def _calculate_technical_signals(self, df: pd.DataFrame) -> float:
        """Расчет технических сигналов"""
        if len(df) < 20:
            return 0.0
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = sma_20 + (bb_std * 2)
        bb_lower = sma_20 - (bb_std * 2)
        
        current_price = df['close'].iloc[-1]
        current_macd = macd.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        # Генерируем сигналы
        signal = 0.0
        
        # RSI сигналы
        if current_rsi < 30:
            signal += 0.3  # Перепроданность
        elif current_rsi > 70:
            signal -= 0.3  # Перекупленность
        
        # MACD сигналы
        if current_macd > current_signal and macd.iloc[-2] <= signal_line.iloc[-2]:
            signal += 0.2  # Золотой крест
        elif current_macd < current_signal and macd.iloc[-2] >= signal_line.iloc[-2]:
            signal -= 0.2  # Мертвый крест
        
        # Bollinger Bands сигналы
        if current_price < bb_lower.iloc[-1]:
            signal += 0.2  # Отскок от нижней полосы
        elif current_price > bb_upper.iloc[-1]:
            signal -= 0.2  # Отскок от верхней полосы
        
        return signal
    
    def backtest_strategy(self, test_data: Dict[str, pd.DataFrame], 
                         news_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Бэктестинг стратегии"""
        logger.info("📊 Начинаем бэктестинг...")
        
        results = {}
        
        for symbol in self.symbols:
            if symbol not in test_data:
                continue
            
            df = test_data[symbol]
            news = news_data.get(symbol, [])
            
            # Рассчитываем сентимент
            sentiment_score = self.sentiment_analyzer.calculate_sentiment_score(news)
            
            # Инициализация
            capital = self.initial_capital
            position = 0
            trades = []
            equity_history = []
            
            # Проходим по данным
            for i in range(60, len(df)):  # Начинаем с 60-го элемента для CNN
                current_data = df.iloc[:i+1]
                current_price = df['close'].iloc[i]
                
                # Генерируем сигнал
                signal = self.generate_trading_signal(symbol, current_data, sentiment_score)
                
                # Выполняем торговлю
                if signal['action'] == 'buy' and position == 0 and signal['confidence'] > 0.5:
                    position = capital / current_price
                    capital = 0
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'time': df['begin'].iloc[i],
                        'confidence': signal['confidence']
                    })
                elif signal['action'] == 'sell' and position > 0 and signal['confidence'] > 0.5:
                    capital = position * current_price
                    position = 0
                    trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'time': df['begin'].iloc[i],
                        'confidence': signal['confidence']
                    })
                
                # Записываем текущую стоимость портфеля
                current_equity = capital + (position * current_price if position > 0 else 0)
                equity_history.append({
                    'time': df['begin'].iloc[i],
                    'equity': current_equity,
                    'price': current_price
                })
            
            # Закрываем позицию
            if position > 0:
                final_price = df['close'].iloc[-1]
                capital = position * final_price
                trades.append({
                    'type': 'sell',
                    'price': final_price,
                    'time': df['begin'].iloc[-1],
                    'confidence': 0.0
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
            volatility = returns.std() * np.sqrt(252 * 24 * 60) * 100
            
            # Sharpe ratio
            risk_free_rate = 0.05
            excess_returns = returns - risk_free_rate / (252 * 24 * 60)
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0
            
            results[symbol] = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': len(trades),
                'trades': trades,
                'equity_history': equity_history,
                'sentiment_score': sentiment_score
            }
            
            logger.info(f"✅ {symbol}: Доходность={total_return:.2f}%, Просадка={max_drawdown:.2f}%, Sharpe={sharpe_ratio:.2f}")
        
        return results

def main():
    """Основная функция"""
    logger.info("🚀 ЗАПУСК ПРОДВИНУТЫХ ТОРГОВЫХ РОБОТОВ")
    logger.info("=" * 60)
    
    # Проверяем доступность библиотек
    if not TENSORTRADE_AVAILABLE:
        logger.warning("⚠️ TensorTrade не доступен, используем упрощенную версию")
    
    if not TENSORFLOW_AVAILABLE:
        logger.warning("⚠️ TensorFlow не доступен, CNN функции будут ограничены")
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("⚠️ Transformers не доступен, анализ новостей будет упрощен")
    
    # Настройки
    symbols = ['GAZP', 'SBER', 'PIKK', 'IRAO', 'SGZH']
    training_days = 365
    test_days = 90
    
    # Инициализация робота
    robot = AdvancedTradingRobot(symbols)
    
    # Загрузка данных
    logger.info("📥 Загружаем исторические данные...")
    all_data = {}
    
    for symbol in symbols:
        df = robot.load_historical_data(symbol, training_days + test_days)
        if not df.empty:
            all_data[symbol] = df
    
    if not all_data:
        logger.error("❌ Не удалось загрузить данные")
        return
    
    # Разделение на обучение и тестирование
    training_data = {}
    test_data = {}
    
    for symbol, df in all_data.items():
        split_point = int(len(df) * (training_days / (training_days + test_days)))
        training_data[symbol] = df.iloc[:split_point]
        test_data[symbol] = df.iloc[split_point:]
    
    # Обучение моделей
    robot.train_all_models(training_data)
    
    # Получение новостей
    logger.info("📰 Анализируем новости...")
    news_data = robot.sentiment_analyzer.get_market_news(symbols)
    
    # Бэктестинг
    results = robot.backtest_strategy(test_data, news_data)
    
    # Сохранение результатов
    with open('advanced_trading_results.json', 'w', encoding='utf-8') as f:
        # Очищаем от циклических ссылок
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() 
                       if k not in ['model', 'scaler', 'anomaly_detector', 'clusterer']}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                return obj
        
        clean_results = clean_for_json(results)
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # Вывод результатов
    logger.info("\n📋 РЕЗУЛЬТАТЫ ПРОДВИНУТЫХ РОБОТОВ:")
    logger.info("=" * 60)
    
    for symbol, result in results.items():
        logger.info(f"{symbol}:")
        logger.info(f"  📈 Доходность: {result['total_return']:.2f}%")
        logger.info(f"  📉 Макс. просадка: {result['max_drawdown']:.2f}%")
        logger.info(f"  📊 Sharpe ratio: {result['sharpe_ratio']:.2f}")
        logger.info(f"  🔄 Количество сделок: {result['total_trades']}")
        logger.info(f"  😊 Сентимент: {result['sentiment_score']:.3f}")
        logger.info("")
    
    # Статистика
    returns = [r['total_return'] for r in results.values()]
    drawdowns = [r['max_drawdown'] for r in results.values()]
    sharpe_ratios = [r['sharpe_ratio'] for r in results.values()]
    
    logger.info("📊 ОБЩАЯ СТАТИСТИКА:")
    logger.info(f"  Средняя доходность: {np.mean(returns):.2f}%")
    logger.info(f"  Средняя просадка: {np.mean(drawdowns):.2f}%")
    logger.info(f"  Средний Sharpe: {np.mean(sharpe_ratios):.2f}")
    
    logger.info(f"\n💾 Результаты сохранены в advanced_trading_results.json")
    logger.info("✅ АНАЛИЗ ЗАВЕРШЕН!")

if __name__ == "__main__":
    main()
