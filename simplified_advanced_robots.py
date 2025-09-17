#!/usr/bin/env python3
"""
Упрощенная версия продвинутых торговых роботов без TensorTrade
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

# ML импорты
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow не установлен")
    # Создаем заглушки для типов
    class Model:
        pass
    class Sequential:
        pass

# NLP импорты
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers не установлен")

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSentimentAnalyzer:
    """Упрощенный анализатор настроений"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipeline = pipeline("sentiment-analysis")
                logger.info("✅ Модель анализа настроений загружена")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки модели: {e}")
    
    def analyze_sentiment(self, text: str) -> float:
        """Анализ настроения текста"""
        if not text or not self.sentiment_pipeline:
            return 0.0
        
        try:
            result = self.sentiment_pipeline(text[:512])
            label = result[0]['label']
            score = result[0]['score']
            
            if 'POSITIVE' in label:
                return score
            elif 'NEGATIVE' in label:
                return -score
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Ошибка анализа настроения: {e}")
            return 0.0
    
    def get_sample_news(self, symbol: str) -> List[Dict]:
        """Получение примеров новостей"""
        return [
            {
                'title': f'{symbol} показывает рост',
                'content': f'Акции {symbol} демонстрируют положительную динамику',
                'sentiment': 0.7
            },
            {
                'title': f'Аналитики оптимистичны по {symbol}',
                'content': f'Эксперты дают позитивные прогнозы по {symbol}',
                'sentiment': 0.8
            }
        ]

class SimpleCNNClassifier:
    """Упрощенный CNN классификатор"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def create_model(self, input_shape: Tuple[int, int] = (60, 5)) -> Model:
        """Создание CNN модели"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен")
        
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')  # up, down, sideways
        ])
        
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных"""
        # Нормализация
        price_data = df[['open', 'high', 'low', 'close', 'volume']].values
        normalized_data = self.scaler.fit_transform(price_data)
        
        # Создание окон
        X, y = [], []
        window_size = 60
        
        for i in range(window_size, len(normalized_data) - 1):
            window = normalized_data[i-window_size:i]
            X.append(window)
            
            # Простая метка направления
            current_price = normalized_data[i, 3]  # close
            next_price = normalized_data[i+1, 3]
            
            if next_price > current_price * 1.01:
                y.append([1, 0, 0])  # up
            elif next_price < current_price * 0.99:
                y.append([0, 1, 0])  # down
            else:
                y.append([0, 0, 1])  # sideways
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, epochs: int = 50):
        """Обучение модели"""
        if not TENSORFLOW_AVAILABLE:
            return
        
        X, y = self.prepare_data(df)
        if len(X) < 100:
            logger.warning("Недостаточно данных для обучения")
            return
        
        self.model = self.create_model()
        
        # Разделение на train/test
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        logger.info(f"✅ CNN обучена. Точность: {history.history['val_accuracy'][-1]:.3f}")
    
    def predict(self, df: pd.DataFrame) -> float:
        """Предсказание направления"""
        if self.model is None:
            return 0.0
        
        try:
            X, _ = self.prepare_data(df)
            if len(X) == 0:
                return 0.0
            
            prediction = self.model.predict(X[-1:], verbose=0)
            direction_probs = prediction[0]
            
            # Возвращаем сигнал: up - down
            return direction_probs[0] - direction_probs[1]
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return 0.0

class SimpleUnsupervisedAgent:
    """Упрощенный агент без учителя"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.clusterer = KMeans(n_clusters=3)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Подготовка признаков"""
        features = []
        
        # Простые индикаторы
        df = df.copy()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['volatility'] = df['close'].rolling(20).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Выбираем признаки
        feature_cols = ['sma_20', 'rsi', 'volatility', 'volume_ratio']
        for col in feature_cols:
            if col in df.columns:
                features.append(df[col].fillna(0).values)
        
        if features:
            return np.column_stack(features)
        return np.array([])
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train(self, df: pd.DataFrame):
        """Обучение"""
        features = self.prepare_features(df)
        if len(features) < 100:
            logger.warning("Недостаточно данных для обучения без учителя")
            return
        
        features_scaled = self.scaler.fit_transform(features)
        
        self.anomaly_detector.fit(features_scaled)
        self.clusterer.fit(features_scaled)
        
        self.is_trained = True
        logger.info("✅ Модели без учителя обучены")
    
    def detect_anomalies(self, df: pd.DataFrame) -> bool:
        """Детекция аномалий"""
        if not self.is_trained:
            return False
        
        features = self.prepare_features(df)
        if len(features) == 0:
            return False
        
        features_scaled = self.scaler.transform(features)
        anomalies = self.anomaly_detector.predict(features_scaled)
        
        return len(anomalies) > 0 and anomalies[-1] == -1

class AdvancedTradingRobot:
    """Продвинутый торговый робот"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 100000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        
        # Компоненты
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.cnn_classifier = SimpleCNNClassifier()
        self.unsupervised_agent = SimpleUnsupervisedAgent()
        
        logger.info("🤖 Продвинутый торговый робот инициализирован")
    
    def load_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Загрузка данных"""
        logger.info(f"📥 Загружаем данные для {symbol}...")
        
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
                        all_data.append(df_batch)
                
                current_date = batch_end
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Ошибка загрузки {symbol}: {e}")
                current_date = batch_end
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('begin').reset_index(drop=True)
            logger.info(f"✅ {symbol}: {len(df)} записей")
            return df
        else:
            logger.error(f"❌ Не удалось загрузить {symbol}")
            return pd.DataFrame()
    
    def train_models(self, training_data: Dict[str, pd.DataFrame]):
        """Обучение всех моделей"""
        logger.info("🎓 Обучение моделей...")
        
        # Объединяем данные для обучения без учителя
        all_data = []
        for df in training_data.values():
            all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self.unsupervised_agent.train(combined_df)
        
        # Обучение CNN для каждого символа
        for symbol, df in training_data.items():
            if len(df) > 200:
                logger.info(f"🎯 Обучение CNN для {symbol}...")
                self.cnn_classifier.train(df)
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация торгового сигнала"""
        try:
            # 1. Анализ новостей
            news = self.sentiment_analyzer.get_sample_news(symbol)
            sentiment_score = np.mean([n['sentiment'] for n in news])
            
            # 2. CNN предсказание
            cnn_signal = self.cnn_classifier.predict(df)
            
            # 3. Детекция аномалий
            anomaly_detected = self.unsupervised_agent.detect_anomalies(df)
            anomaly_signal = 0.3 if anomaly_detected else 0.0
            
            # 4. Технические индикаторы
            technical_signal = self._calculate_technical_signals(df)
            
            # 5. Объединяем сигналы
            final_signal = (
                sentiment_score * 0.3 +
                cnn_signal * 0.4 +
                technical_signal * 0.2 +
                anomaly_signal * 0.1
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
                    'cnn': cnn_signal,
                    'technical': technical_signal,
                    'anomaly': anomaly_signal
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации сигнала для {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'final_signal': 0.0}
    
    def _calculate_technical_signals(self, df: pd.DataFrame) -> float:
        """Технические сигналы"""
        if len(df) < 20:
            return 0.0
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Простые сигналы
        signal = 0.0
        
        if current_rsi < 30:
            signal += 0.3  # Перепроданность
        elif current_rsi > 70:
            signal -= 0.3  # Перекупленность
        
        # Тренд
        sma_20 = df['close'].rolling(20).mean()
        if df['close'].iloc[-1] > sma_20.iloc[-1]:
            signal += 0.2
        else:
            signal -= 0.2
        
        return signal
    
    def backtest(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Бэктестинг"""
        logger.info("📊 Бэктестинг...")
        
        results = {}
        
        for symbol, df in test_data.items():
            capital = self.initial_capital
            position = 0
            trades = []
            equity_history = []
            
            for i in range(60, len(df)):
                current_data = df.iloc[:i+1]
                current_price = df['close'].iloc[i]
                
                signal = self.generate_signal(symbol, current_data)
                
                # Торговля
                if signal['action'] == 'buy' and position == 0 and signal['confidence'] > 0.5:
                    position = capital / current_price
                    capital = 0
                    trades.append({'type': 'buy', 'price': current_price, 'time': df['begin'].iloc[i]})
                elif signal['action'] == 'sell' and position > 0 and signal['confidence'] > 0.5:
                    capital = position * current_price
                    position = 0
                    trades.append({'type': 'sell', 'price': current_price, 'time': df['begin'].iloc[i]})
                
                # Записываем equity
                current_equity = capital + (position * current_price if position > 0 else 0)
                equity_history.append({'time': df['begin'].iloc[i], 'equity': current_equity})
            
            # Закрываем позицию
            if position > 0:
                final_price = df['close'].iloc[-1]
                capital = position * final_price
            
            # Метрики
            final_equity = capital
            total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
            
            # Просадка
            equity_series = pd.Series([h['equity'] for h in equity_history])
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
            
            # Волатильность
            returns = equity_series.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252 * 24 * 60) * 100
            
            # Sharpe
            risk_free_rate = 0.05
            excess_returns = returns - risk_free_rate / (252 * 24 * 60)
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0
            
            results[symbol] = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': len(trades)
            }
            
            logger.info(f"✅ {symbol}: Доходность={total_return:.2f}%, Просадка={max_drawdown:.2f}%, Sharpe={sharpe_ratio:.2f}")
        
        return results

def main():
    """Основная функция"""
    logger.info("🚀 ЗАПУСК УПРОЩЕННЫХ ПРОДВИНУТЫХ РОБОТОВ")
    logger.info("=" * 60)
    
    symbols = ['GAZP', 'SBER', 'PIKK', 'IRAO', 'SGZH']
    training_days = 365
    test_days = 90
    
    robot = AdvancedTradingRobot(symbols)
    
    # Загрузка данных
    logger.info("📥 Загружаем данные...")
    all_data = {}
    
    for symbol in symbols:
        df = robot.load_data(symbol, training_days + test_days)
        if not df.empty:
            all_data[symbol] = df
    
    if not all_data:
        logger.error("❌ Не удалось загрузить данные")
        return
    
    # Разделение данных
    training_data = {}
    test_data = {}
    
    for symbol, df in all_data.items():
        split_point = int(len(df) * (training_days / (training_days + test_days)))
        training_data[symbol] = df.iloc[:split_point]
        test_data[symbol] = df.iloc[split_point:]
    
    # Обучение
    robot.train_models(training_data)
    
    # Бэктестинг
    results = robot.backtest(test_data)
    
    # Сохранение
    with open('simplified_advanced_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Вывод результатов
    logger.info("\n📋 РЕЗУЛЬТАТЫ:")
    logger.info("=" * 60)
    
    for symbol, result in results.items():
        logger.info(f"{symbol}:")
        logger.info(f"  📈 Доходность: {result['total_return']:.2f}%")
        logger.info(f"  📉 Просадка: {result['max_drawdown']:.2f}%")
        logger.info(f"  📊 Sharpe: {result['sharpe_ratio']:.2f}")
        logger.info(f"  🔄 Сделки: {result['total_trades']}")
        logger.info("")
    
    # Статистика
    returns = [r['total_return'] for r in results.values()]
    logger.info(f"📊 Средняя доходность: {np.mean(returns):.2f}%")
    
    logger.info("✅ АНАЛИЗ ЗАВЕРШЕН!")

if __name__ == "__main__":
    main()
