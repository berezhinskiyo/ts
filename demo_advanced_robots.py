#!/usr/bin/env python3
"""
Демо версия продвинутых торговых роботов без внешних зависимостей
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

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoSentimentAnalyzer:
    """Демо анализатор настроений"""
    
    def __init__(self):
        # Словари для простого анализа настроений
        self.positive_words = [
            'рост', 'повышение', 'увеличение', 'позитивный', 'хороший', 'отличный',
            'успех', 'прибыль', 'доход', 'выигрыш', 'победа', 'улучшение'
        ]
        self.negative_words = [
            'падение', 'снижение', 'уменьшение', 'негативный', 'плохой', 'убыток',
            'потеря', 'провал', 'кризис', 'проблема', 'риск', 'опасность'
        ]
        logger.info("✅ Демо анализатор настроений инициализирован")
    
    def analyze_sentiment(self, text: str) -> float:
        """Простой анализ настроения по ключевым словам"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Нормализуем результат
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment * 10))  # Масштабируем и ограничиваем
    
    def get_sample_news(self, symbol: str) -> List[Dict]:
        """Получение примеров новостей"""
        return [
            {
                'title': f'{symbol} показывает устойчивый рост на фоне позитивных отчетов',
                'content': f'Акции {symbol} демонстрируют отличные результаты благодаря улучшению финансовых показателей и успешной стратегии развития',
                'sentiment': 0.7
            },
            {
                'title': f'Аналитики повышают прогнозы по {symbol}',
                'content': f'Ведущие эксперты пересматривают свои прогнозы по {symbol} в сторону повышения, ожидая дальнейшего роста',
                'sentiment': 0.8
            },
            {
                'title': f'Рынок проявляет осторожность по {symbol}',
                'content': f'Инвесторы проявляют осторожность в отношении {symbol} из-за неопределенности на рынке',
                'sentiment': -0.3
            }
        ]
    
    def calculate_sentiment_score(self, news_list: List[Dict]) -> float:
        """Расчет общего индекса настроений"""
        if not news_list:
            return 0.0
        
        # Используем предустановленные значения или анализируем текст
        sentiments = []
        for news in news_list:
            if 'sentiment' in news:
                sentiments.append(news['sentiment'])
            else:
                # Анализируем текст
                title_sentiment = self.analyze_sentiment(news.get('title', ''))
                content_sentiment = self.analyze_sentiment(news.get('content', ''))
                combined_sentiment = (title_sentiment * 0.7 + content_sentiment * 0.3)
                sentiments.append(combined_sentiment)
        
        return np.mean(sentiments)

class DemoPatternClassifier:
    """Демо классификатор паттернов"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.pattern_types = ['bullish', 'bearish', 'sideways']
        logger.info("✅ Демо классификатор паттернов инициализирован")
    
    def create_features(self, df: pd.DataFrame) -> np.ndarray:
        """Создание признаков для классификации паттернов"""
        features = []
        
        # Технические индикаторы
        df = df.copy()
        
        # Скользящие средние
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
        
        # Заполняем NaN и выбираем последние значения
        for col in feature_columns:
            if col in df.columns:
                features.append(df[col].fillna(0).iloc[-1])
        
        return np.array(features)
    
    def train(self, df: pd.DataFrame):
        """Обучение модели классификации паттернов"""
        logger.info("🎯 Обучение демо классификатора паттернов...")
        
        # Подготавливаем данные
        X = []
        y = []
        
        window_size = 20
        
        for i in range(window_size, len(df) - 1):
            # Создаем окно данных
            window_df = df.iloc[i-window_size:i+1]
            features = self.create_features(window_df)
            
            if len(features) > 0:
                X.append(features)
                
                # Определяем паттерн на основе будущего движения
                current_price = df['close'].iloc[i]
                future_price = df['close'].iloc[i+1]
                price_change = (future_price - current_price) / current_price
                
                if price_change > 0.02:  # Рост > 2%
                    y.append(0)  # bullish
                elif price_change < -0.02:  # Падение > 2%
                    y.append(1)  # bearish
                else:
                    y.append(2)  # sideways
        
        if len(X) < 50:
            logger.warning("Недостаточно данных для обучения")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Нормализация
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучение Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        logger.info("✅ Демо классификатор паттернов обучен")
    
    def predict_pattern(self, df: pd.DataFrame) -> float:
        """Предсказание паттерна"""
        if self.model is None:
            return 0.0
        
        try:
            features = self.create_features(df)
            if len(features) == 0:
                return 0.0
            
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            
            # Преобразуем в сигнал: bullish=1, bearish=-1, sideways=0
            if prediction < 0.5:
                return 1.0  # bullish
            elif prediction > 1.5:
                return -1.0  # bearish
            else:
                return 0.0  # sideways
                
        except Exception as e:
            logger.error(f"Ошибка предсказания паттерна: {e}")
            return 0.0

class DemoUnsupervisedAgent:
    """Демо агент для обучения без учителя"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        self.pca = PCA(n_components=5)
        self.scaler = StandardScaler()
        self.is_trained = False
        logger.info("✅ Демо агент без учителя инициализирован")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Подготовка признаков"""
        features = []
        
        # Простые индикаторы
        df = df.copy()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['volatility'] = df['close'].rolling(20).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_change'] = df['close'].pct_change()
        
        # Выбираем признаки
        feature_cols = ['sma_20', 'rsi', 'volatility', 'volume_ratio', 'price_change']
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
        """Обучение моделей без учителя"""
        logger.info("🤖 Обучение демо моделей без учителя...")
        
        features = self.prepare_features(df)
        if len(features) < 100:
            logger.warning("Недостаточно данных для обучения без учителя")
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
        logger.info("✅ Демо модели без учителя обучены")
    
    def detect_anomalies(self, df: pd.DataFrame) -> bool:
        """Детекция аномалий"""
        if not self.is_trained:
            return False
        
        features = self.prepare_features(df)
        if len(features) == 0:
            return False
        
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        anomalies = self.anomaly_detector.predict(features_pca)
        return len(anomalies) > 0 and anomalies[-1] == -1
    
    def get_market_regime(self, df: pd.DataFrame) -> int:
        """Определение рыночного режима"""
        if not self.is_trained:
            return 0
        
        features = self.prepare_features(df)
        if len(features) == 0:
            return 0
        
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        cluster = self.clusterer.predict(features_pca)
        return cluster[-1] if len(cluster) > 0 else 0

class AdvancedTradingRobot:
    """Продвинутый торговый робот с демо компонентами"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 100000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        
        # Компоненты
        self.sentiment_analyzer = DemoSentimentAnalyzer()
        self.pattern_classifier = DemoPatternClassifier()
        self.unsupervised_agent = DemoUnsupervisedAgent()
        
        # Веса для объединения сигналов
        self.sentiment_weight = 0.25
        self.pattern_weight = 0.35
        self.technical_weight = 0.25
        self.anomaly_weight = 0.15
        
        logger.info("🤖 Продвинутый торговый робот с демо компонентами инициализирован")
    
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
        logger.info("🎓 Обучение всех демо моделей...")
        
        # Объединяем данные для обучения без учителя
        all_data = []
        for df in training_data.values():
            all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self.unsupervised_agent.train(combined_df)
        
        # Обучение классификатора паттернов для каждого символа
        for symbol, df in training_data.items():
            if len(df) > 100:
                logger.info(f"🎯 Обучение классификатора паттернов для {symbol}...")
                self.pattern_classifier.train(df)
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация торгового сигнала"""
        try:
            # 1. Анализ настроений новостей
            news = self.sentiment_analyzer.get_sample_news(symbol)
            sentiment_score = self.sentiment_analyzer.calculate_sentiment_score(news)
            
            # 2. Классификация паттернов
            pattern_signal = self.pattern_classifier.predict_pattern(df)
            
            # 3. Детекция аномалий
            anomaly_detected = self.unsupervised_agent.detect_anomalies(df)
            anomaly_signal = 0.5 if anomaly_detected else 0.0
            
            # 4. Рыночный режим
            market_regime = self.unsupervised_agent.get_market_regime(df)
            regime_signal = (market_regime - 2) * 0.2  # Центрируем вокруг 0
            
            # 5. Технические индикаторы
            technical_signal = self._calculate_technical_signals(df)
            
            # 6. Объединяем все сигналы
            final_signal = (
                sentiment_score * self.sentiment_weight +
                pattern_signal * self.pattern_weight +
                technical_signal * self.technical_weight +
                anomaly_signal * self.anomaly_weight +
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
        
        signal = 0.0
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
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
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()
        
        if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
            signal += 0.2  # Золотой крест
        elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
            signal -= 0.2  # Мертвый крест
        
        return signal
    
    def backtest(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Бэктестинг стратегии"""
        logger.info("📊 Бэктестинг продвинутой стратегии...")
        
        results = {}
        
        for symbol, df in test_data.items():
            capital = self.initial_capital
            position = 0
            trades = []
            equity_history = []
            
            for i in range(60, len(df)):  # Начинаем с 60-го элемента
                current_data = df.iloc[:i+1]
                current_price = df['close'].iloc[i]
                
                signal = self.generate_signal(symbol, current_data)
                
                # Торговля
                if signal['action'] == 'buy' and position == 0 and signal['confidence'] > 0.4:
                    position = capital / current_price
                    capital = 0
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'time': df['begin'].iloc[i],
                        'confidence': signal['confidence']
                    })
                elif signal['action'] == 'sell' and position > 0 and signal['confidence'] > 0.4:
                    capital = position * current_price
                    position = 0
                    trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'time': df['begin'].iloc[i],
                        'confidence': signal['confidence']
                    })
                
                # Записываем equity
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
            
            results[symbol] = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'trades': trades,
                'equity_history': equity_history
            }
            
            logger.info(f"✅ {symbol}: Доходность={total_return:.2f}%, Просадка={max_drawdown:.2f}%, Sharpe={sharpe_ratio:.2f}, Win Rate={win_rate:.1f}%")
        
        return results

def main():
    """Основная функция"""
    logger.info("🚀 ЗАПУСК ДЕМО ПРОДВИНУТЫХ ТОРГОВЫХ РОБОТОВ")
    logger.info("=" * 60)
    
    symbols = ['GAZP', 'SBER', 'PIKK', 'IRAO', 'SGZH']
    training_days = 365
    test_days = 90
    
    robot = AdvancedTradingRobot(symbols)
    
    # Загрузка данных
    logger.info("📥 Загружаем исторические данные...")
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
    
    # Обучение моделей
    robot.train_models(training_data)
    
    # Бэктестинг
    results = robot.backtest(test_data)
    
    # Сохранение результатов
    with open('demo_advanced_results.json', 'w', encoding='utf-8') as f:
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
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
    logger.info("\n📋 РЕЗУЛЬТАТЫ ДЕМО ПРОДВИНУТЫХ РОБОТОВ:")
    logger.info("=" * 60)
    
    summary_data = []
    for symbol, result in results.items():
        summary_data.append({
            'Symbol': symbol,
            'Return': f"{result['total_return']:.2f}%",
            'Drawdown': f"{result['max_drawdown']:.2f}%",
            'Sharpe': f"{result['sharpe_ratio']:.2f}",
            'Win_Rate': f"{result['win_rate']:.1f}%",
            'Trades': result['total_trades']
        })
        
        logger.info(f"{symbol}:")
        logger.info(f"  📈 Доходность: {result['total_return']:.2f}%")
        logger.info(f"  📉 Макс. просадка: {result['max_drawdown']:.2f}%")
        logger.info(f"  📊 Sharpe ratio: {result['sharpe_ratio']:.2f}")
        logger.info(f"  🎯 Win rate: {result['win_rate']:.1f}%")
        logger.info(f"  🔄 Количество сделок: {result['total_trades']}")
        logger.info("")
    
    # Создаем сводную таблицу
    results_df = pd.DataFrame(summary_data)
    results_df.to_csv('demo_advanced_summary.csv', index=False)
    
    # Статистика
    returns = [r['total_return'] for r in results.values()]
    drawdowns = [r['max_drawdown'] for r in results.values()]
    sharpe_ratios = [r['sharpe_ratio'] for r in results.values()]
    win_rates = [r['win_rate'] for r in results.values()]
    
    logger.info("📊 ОБЩАЯ СТАТИСТИКА:")
    logger.info(f"  Средняя доходность: {np.mean(returns):.2f}%")
    logger.info(f"  Средняя просадка: {np.mean(drawdowns):.2f}%")
    logger.info(f"  Средний Sharpe: {np.mean(sharpe_ratios):.2f}")
    logger.info(f"  Средний Win Rate: {np.mean(win_rates):.1f}%")
    
    logger.info(f"\n💾 Результаты сохранены:")
    logger.info(f"  📄 Детали: demo_advanced_results.json")
    logger.info(f"  📄 Сводка: demo_advanced_summary.csv")
    
    logger.info("\n✅ ДЕМО АНАЛИЗ ЗАВЕРШЕН!")

if __name__ == "__main__":
    main()
