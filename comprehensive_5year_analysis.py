#!/usr/bin/env python3
"""
Комплексный анализ за 5 лет: загрузка данных, обучение моделей и тестирование стратегий
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('5year_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataDownloader:
    """Класс для загрузки исторических данных с MOEX"""
    
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss/"
        self.data_dir = "data/5year_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_instrument_data(self, symbol: str, years: int = 5) -> pd.DataFrame:
        """Загрузка данных за указанное количество лет"""
        logger.info(f"📥 Загружаем данные для {symbol} за {years} лет...")
        
        all_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Разбиваем на части по 90 дней (из-за лимита 500 записей)
        current_date = start_date
        batch_size = 90  # дней
        
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
                time.sleep(0.5)  # Пауза между запросами
                
            except Exception as e:
                logger.error(f"  ❌ {symbol}: Ошибка за {current_date.strftime('%Y-%m-%d')}: {e}")
                current_date = batch_end
                time.sleep(1)
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('begin').reset_index(drop=True)
            
            # Сохраняем данные
            filename = f"{self.data_dir}/{symbol}_5year.csv"
            df.to_csv(filename, index=False)
            logger.info(f"💾 {symbol}: Сохранено {len(df)} записей в {filename}")
            
            return df
        else:
            logger.error(f"❌ {symbol}: Не удалось загрузить данные")
            return pd.DataFrame()
    
    def download_all_instruments(self, symbols: List[str], years: int = 5) -> Dict[str, pd.DataFrame]:
        """Загрузка данных для всех инструментов"""
        logger.info(f"🚀 Начинаем загрузку данных для {len(symbols)} инструментов за {years} лет")
        
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
    """Класс для расчета технических индикаторов"""
    
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
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание всех технических индикаторов"""
        df = df.copy()
        
        # Основные индикаторы
        df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        df['ema_12'] = TechnicalIndicators.ema(df['close'], 12)
        df['ema_26'] = TechnicalIndicators.ema(df['close'], 26)
        
        # RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'])
        
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
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Дополнительные признаки
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Временные признаки
        df['hour'] = df['begin'].dt.hour
        df['day_of_week'] = df['begin'].dt.dayofweek
        df['month'] = df['begin'].dt.month
        
        return df

class MLModelTrainer:
    """Класс для обучения ML моделей"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'future_return', 
                    lookback: int = 20, forecast_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        # Создаем целевую переменную
        df['future_price'] = df['close'].shift(-forecast_horizon)
        df['future_return'] = (df['future_price'] / df['close'] - 1) * 100
        
        # Убираем NaN
        df = df.dropna()
        
        # Выбираем признаки
        feature_columns = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
            'bb_width', 'atr', 'price_change', 'volume_change', 'high_low_ratio',
            'close_open_ratio', 'hour', 'day_of_week', 'month'
        ]
        
        # Создаем скользящие окна
        X = []
        y = []
        
        for i in range(lookback, len(df) - forecast_horizon):
            window_data = df[feature_columns].iloc[i-lookback:i].values
            target_value = df[target_column].iloc[i]
            
            if not np.isnan(target_value):
                X.append(window_data.flatten())
                y.append(target_value)
        
        return np.array(X), np.array(y)
    
    def train_models(self, X: np.ndarray, y: np.ndarray, symbol: str):
        """Обучение различных ML моделей"""
        logger.info(f"🤖 Обучаем модели для {symbol}...")
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Модели для обучения
        models_config = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        results = {}
        
        for name, model in models_config.items():
            try:
                logger.info(f"  🔄 Обучаем {name}...")
                
                # Обучение
                if name in ['ridge', 'svr']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Оценка
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                logger.info(f"    ✅ {name}: MSE={mse:.4f}, R²={r2:.4f}")
                
            except Exception as e:
                logger.error(f"    ❌ Ошибка обучения {name}: {e}")
        
        # Сохраняем результаты
        self.models[symbol] = results
        self.scalers[symbol] = scaler
        
        return results

class StrategyTester:
    """Класс для тестирования торговых стратегий"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
    
    def test_ml_strategy(self, df: pd.DataFrame, models: Dict, symbol: str) -> Dict:
        """Тестирование ML стратегии"""
        logger.info(f"📊 Тестируем ML стратегию для {symbol}...")
        
        capital = self.initial_capital
        position = 0
        trades = []
        equity_history = []
        
        # Параметры стратегии
        lookback = 20
        forecast_horizon = 5
        confidence_threshold = 0.5  # Минимальная уверенность для входа
        
        for i in range(lookback, len(df) - forecast_horizon):
            current_price = df['close'].iloc[i]
            
            # Подготавливаем данные для предсказания
            feature_columns = [
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
                'bb_width', 'atr', 'price_change', 'volume_change', 'high_low_ratio',
                'close_open_ratio', 'hour', 'day_of_week', 'month'
            ]
            
            window_data = df[feature_columns].iloc[i-lookback:i].values.flatten()
            
            # Получаем предсказания от всех моделей
            predictions = []
            for model_name, model_data in models.items():
                try:
                    if model_name in ['ridge', 'svr']:
                        # Нужна нормализация
                        window_scaled = models['scaler'].transform([window_data])
                        pred = model_data['model'].predict(window_scaled)[0]
                    else:
                        pred = model_data['model'].predict([window_data])[0]
                    predictions.append(pred)
                except:
                    continue
            
            if not predictions:
                continue
            
            # Усредняем предсказания
            avg_prediction = np.mean(predictions)
            confidence = 1.0 - (np.std(predictions) / (abs(avg_prediction) + 1e-6))
            
            # Логика торговли
            if confidence > confidence_threshold:
                if avg_prediction > 0.5 and position == 0:  # Покупка
                    position = capital / current_price
                    capital = 0
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'time': df['begin'].iloc[i],
                        'prediction': avg_prediction,
                        'confidence': confidence
                    })
                elif avg_prediction < -0.5 and position > 0:  # Продажа
                    capital = position * current_price
                    position = 0
                    trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'time': df['begin'].iloc[i],
                        'prediction': avg_prediction,
                        'confidence': confidence
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
                'confidence': 0
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
            'trades': trades,
            'equity_history': equity_history
        }
        
        logger.info(f"  ✅ {symbol}: Доходность={total_return:.2f}%, Просадка={max_drawdown:.2f}%, Sharpe={sharpe_ratio:.2f}")
        
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
    logger.info("🚀 ЗАПУСК КОМПЛЕКСНОГО АНАЛИЗА ЗА 5 ЛЕТ")
    logger.info("=" * 60)
    
    # Инструменты для анализа
    symbols = ['GAZP', 'SBER', 'PIKK', 'IRAO', 'SGZH']
    
    # 1. Загрузка данных
    logger.info("📥 ЭТАП 1: Загрузка исторических данных")
    downloader = DataDownloader()
    all_data = downloader.download_all_instruments(symbols, years=5)
    
    if not all_data:
        logger.error("❌ Не удалось загрузить данные. Завершение работы.")
        return
    
    # 2. Подготовка данных и обучение моделей
    logger.info("\n🤖 ЭТАП 2: Обучение ML моделей")
    trainer = MLModelTrainer()
    indicators = TechnicalIndicators()
    
    trained_models = {}
    
    for symbol, df in all_data.items():
        logger.info(f"📊 Обрабатываем {symbol}...")
        
        # Создаем технические индикаторы
        df_with_features = indicators.create_features(df)
        
        # Подготавливаем данные для обучения
        X, y = trainer.prepare_data(df_with_features)
        
        if len(X) > 100:  # Минимум данных для обучения
            # Обучаем модели
            models = trainer.train_models(X, y, symbol)
            trained_models[symbol] = {
                'models': models,
                'data': df_with_features
            }
        else:
            logger.warning(f"⚠️ {symbol}: Недостаточно данных для обучения ({len(X)} записей)")
    
    # 3. Тестирование стратегий
    logger.info("\n📊 ЭТАП 3: Тестирование торговых стратегий")
    tester = StrategyTester()
    
    all_results = {}
    
    for symbol, model_data in trained_models.items():
        logger.info(f"🧪 Тестируем стратегии для {symbol}...")
        
        df = model_data['data']
        models = model_data['models']
        
        # Тестируем ML стратегию
        ml_result = tester.test_ml_strategy(df, models, symbol)
        
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
            'ML_Trades': ml['total_trades'],
            'BH_Return': f"{bh['total_return']:.2f}%",
            'BH_Drawdown': f"{bh['max_drawdown']:.2f}%",
            'BH_Sharpe': f"{bh['sharpe_ratio']:.2f}",
            'Outperformance': f"{ml['total_return'] - bh['total_return']:.2f}%"
        })
    
    # Сохраняем результаты
    results_df = pd.DataFrame(summary_data)
    results_df.to_csv('5year_analysis_results.csv', index=False)
    
    # Сохраняем детальные результаты (без циклических ссылок)
    def clean_for_json(obj):
        """Очистка объекта от циклических ссылок и numpy типов"""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if k not in ['model', 'scaler']}
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
    
    with open('5year_detailed_results.json', 'w', encoding='utf-8') as f:
        clean_results = clean_for_json(all_results)
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # Выводим сводку
    logger.info("\n📋 СВОДКА РЕЗУЛЬТАТОВ:")
    logger.info("=" * 60)
    print(results_df.to_string(index=False))
    
    # Статистика
    ml_returns = [float(r['ML_Return'].replace('%', '')) for r in summary_data]
    bh_returns = [float(r['BH_Return'].replace('%', '')) for r in summary_data]
    
    logger.info(f"\n📊 СТАТИСТИКА:")
    logger.info(f"  Средняя доходность ML: {np.mean(ml_returns):.2f}%")
    logger.info(f"  Средняя доходность Buy & Hold: {np.mean(bh_returns):.2f}%")
    logger.info(f"  Превышение ML над Buy & Hold: {np.mean(ml_returns) - np.mean(bh_returns):.2f}%")
    
    logger.info(f"\n💾 Результаты сохранены:")
    logger.info(f"  📄 Сводка: 5year_analysis_results.csv")
    logger.info(f"  📄 Детали: 5year_detailed_results.json")
    logger.info(f"  📄 Логи: 5year_analysis.log")
    
    logger.info("\n✅ АНАЛИЗ ЗАВЕРШЕН!")

if __name__ == "__main__":
    main()
