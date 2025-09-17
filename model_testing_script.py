#!/usr/bin/env python3
"""
Скрипт для тестирования обученных ML моделей
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTester:
    """Класс для тестирования обученных моделей"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.models_dir = "trained_models"
        self.unsupervised_dir = "unsupervised_models"
        self.results = {}
    
    def load_models(self, symbol: str) -> Dict:
        """Загрузка обученных моделей для символа"""
        symbol_dir = os.path.join(self.models_dir, symbol)
        
        if not os.path.exists(symbol_dir):
            logger.warning(f"Модели для {symbol} не найдены")
            return {}
        
        models = {}
        
        # Загружаем каждую модель
        for model_name in ['random_forest', 'gradient_boosting', 'ridge', 'linear_regression']:
            model_path = os.path.join(symbol_dir, f"{model_name}.joblib")
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        
        # Загружаем scaler и feature_selector
        scaler_path = os.path.join(symbol_dir, "scaler.joblib")
        selector_path = os.path.join(symbol_dir, "feature_selector.joblib")
        
        scaler = None
        feature_selector = None
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        if os.path.exists(selector_path):
            feature_selector = joblib.load(selector_path)
        
        return {
            'models': models,
            'scaler': scaler,
            'feature_selector': feature_selector
        }
    
    def load_unsupervised_models(self) -> Dict:
        """Загрузка моделей без учителя"""
        models = {}
        
        try:
            models['anomaly_detector'] = joblib.load(os.path.join(self.unsupervised_dir, "anomaly_detector.joblib"))
            models['clusterer'] = joblib.load(os.path.join(self.unsupervised_dir, "clusterer.joblib"))
            models['pca'] = joblib.load(os.path.join(self.unsupervised_dir, "pca.joblib"))
            models['scaler'] = joblib.load(os.path.join(self.unsupervised_dir, "scaler.joblib"))
            logger.info("✅ Модели без учителя загружены")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить модели без учителя: {e}")
        
        return models
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Подготовка признаков для предсказания"""
        feature_columns = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr', 'price_change', 'volume_change', 'high_low_ratio', 'close_open_ratio',
            'body_size', 'volatility_5', 'volatility_20', 'hour', 'day_of_week', 'month',
            'volume_sma', 'volume_ratio'
        ]
        
        # Фильтруем доступные колонки
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_columns:
            return np.array([])
        
        features = df[available_columns].fillna(0).values
        return features
    
    def generate_signal(self, df: pd.DataFrame, symbol: str, model_data: Dict, 
                       unsupervised_models: Dict) -> Dict[str, any]:
        """Генерация торгового сигнала"""
        try:
            # Подготавливаем данные для предсказания
            lookback = 30
            if len(df) < lookback:
                return {'action': 'hold', 'confidence': 0.0, 'final_signal': 0.0}
            
            # Получаем последние данные
            recent_data = df.tail(lookback)
            features = self.prepare_features(recent_data)
            
            if len(features) == 0:
                return {'action': 'hold', 'confidence': 0.0, 'final_signal': 0.0}
            
            # Получаем предсказания от всех моделей
            predictions = []
            model_weights = []
            
            for model_name, model in model_data['models'].items():
                try:
                    if model_name in ['ridge', 'linear_regression']:
                        # Нужна нормализация и выбор признаков
                        if model_data['scaler'] and model_data['feature_selector']:
                            features_scaled = model_data['scaler'].transform(features)
                            features_selected = model_data['feature_selector'].transform(features_scaled)
                            pred = model.predict(features_selected)[0]
                        else:
                            pred = model.predict(features)[0]
                    else:
                        pred = model.predict(features)[0]
                    
                    predictions.append(pred)
                    # Простой вес модели
                    model_weights.append(1.0)
                    
                except Exception as e:
                    logger.error(f"Ошибка предсказания {model_name}: {e}")
                    continue
            
            if not predictions:
                return {'action': 'hold', 'confidence': 0.0, 'final_signal': 0.0}
            
            # Взвешенное усреднение предсказаний
            avg_prediction = np.average(predictions, weights=model_weights)
            confidence = 1.0 - (np.std(predictions) / (abs(avg_prediction) + 1e-6))
            
            # Анализ аномалий и режимов
            anomaly_signal = 0.0
            regime_signal = 0.0
            
            if unsupervised_models:
                try:
                    # Подготавливаем признаки для моделей без учителя
                    unsupervised_features = self.prepare_features_for_unsupervised(df.tail(100))
                    
                    if len(unsupervised_features) > 0:
                        # Детекция аномалий
                        if 'scaler' in unsupervised_models and 'pca' in unsupervised_models and 'anomaly_detector' in unsupervised_models:
                            features_scaled = unsupervised_models['scaler'].transform(unsupervised_features)
                            features_pca = unsupervised_models['pca'].transform(features_scaled)
                            anomalies = unsupervised_models['anomaly_detector'].predict(features_pca)
                            
                            if len(anomalies) > 0 and anomalies[-1] == -1:
                                anomaly_signal = 0.3
                        
                        # Определение режимов
                        if 'scaler' in unsupervised_models and 'pca' in unsupervised_models and 'clusterer' in unsupervised_models:
                            features_scaled = unsupervised_models['scaler'].transform(unsupervised_features)
                            features_pca = unsupervised_models['pca'].transform(features_scaled)
                            clusters = unsupervised_models['clusterer'].predict(features_pca)
                            
                            if len(clusters) > 0:
                                current_regime = clusters[-1]
                                regime_signal = (current_regime - 2) * 0.1
                
                except Exception as e:
                    logger.error(f"Ошибка анализа без учителя: {e}")
            
            # Объединяем сигналы
            final_signal = avg_prediction + anomaly_signal + regime_signal
            
            # Определяем действие
            if final_signal > 0.5:
                action = 'buy'
                confidence = min(final_signal, 1.0)
            elif final_signal < -0.5:
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
                    'prediction': avg_prediction,
                    'anomaly': anomaly_signal,
                    'regime': regime_signal
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации сигнала для {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'final_signal': 0.0}
    
    def prepare_features_for_unsupervised(self, df: pd.DataFrame) -> np.ndarray:
        """Подготовка признаков для моделей без учителя"""
        feature_columns = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_width', 'atr',
            'volatility_20', 'volume_ratio', 'price_change', 'high_low_ratio'
        ]
        
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_columns:
            return np.array([])
        
        features = df[available_columns].fillna(0).values
        return features
    
    def test_strategy(self, df: pd.DataFrame, symbol: str, model_data: Dict, 
                     unsupervised_models: Dict) -> Dict:
        """Тестирование торговой стратегии"""
        logger.info(f"📊 Тестируем стратегию для {symbol}...")
        
        capital = self.initial_capital
        position = 0
        trades = []
        equity_history = []
        
        # Параметры стратегии
        confidence_threshold = 0.6
        
        for i in range(60, len(df)):  # Начинаем с 60-го элемента
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            
            # Генерируем сигнал
            signal = self.generate_signal(current_data, symbol, model_data, unsupervised_models)
            
            # Выполняем торговлю
            if signal['action'] == 'buy' and position == 0 and signal['confidence'] > confidence_threshold:
                position = capital / current_price
                capital = 0
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'time': df['begin'].iloc[i],
                    'confidence': signal['confidence'],
                    'signal': signal['final_signal']
                })
            elif signal['action'] == 'sell' and position > 0 and signal['confidence'] > confidence_threshold:
                capital = position * current_price
                position = 0
                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'time': df['begin'].iloc[i],
                    'confidence': signal['confidence'],
                    'signal': signal['final_signal']
                })
            
            # Записываем текущую стоимость портфеля
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
                'confidence': 0.0,
                'signal': 0.0
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

def load_data_from_files(symbols: List[str], data_dir: str = "data/3year_minute_data") -> Dict[str, pd.DataFrame]:
    """Загрузка данных из файлов"""
    logger.info("📥 Загружаем данные из файлов...")
    
    all_data = {}
    
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_3year_minute.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['begin'] = pd.to_datetime(df['begin'])
                df['end'] = pd.to_datetime(df['end'])
                
                # Удаляем дубликаты
                df = df.drop_duplicates(subset=['begin']).reset_index(drop=True)
                
                all_data[symbol] = df
                logger.info(f"✅ {symbol}: Загружено {len(df)} записей")
                
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {symbol}: {e}")
        else:
            logger.warning(f"⚠️ Файл {file_path} не найден")
    
    return all_data

def main():
    """Основная функция тестирования моделей"""
    logger.info("🚀 ЗАПУСК ТЕСТИРОВАНИЯ ML МОДЕЛЕЙ")
    logger.info("=" * 50)
    
    # Инструменты для тестирования
    symbols = ['GAZP', 'SBER', 'PIKK', 'IRAO', 'SGZH']
    
    # Загрузка данных
    all_data = load_data_from_files(symbols)
    
    if not all_data:
        logger.error("❌ Не удалось загрузить данные. Завершение работы.")
        return
    
    # Инициализация тестера
    tester = ModelTester()
    
    # Загрузка моделей без учителя
    unsupervised_models = tester.load_unsupervised_models()
    
    # Тестирование стратегий
    all_results = {}
    
    for symbol, df in all_data.items():
        logger.info(f"🧪 Тестируем стратегии для {symbol}...")
        
        # Загрузка моделей для символа
        model_data = tester.load_models(symbol)
        
        if not model_data['models']:
            logger.warning(f"⚠️ Модели для {symbol} не найдены, пропускаем")
            continue
        
        # Тестируем ML стратегию
        ml_result = tester.test_strategy(df, symbol, model_data, unsupervised_models)
        
        # Тестируем Buy & Hold
        bh_result = tester.test_buy_hold_strategy(df, symbol)
        
        all_results[symbol] = {
            'ml_strategy': ml_result,
            'buy_hold': bh_result
        }
    
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
            'BH_Return': f"{bh['total_return']:.2f}%",
            'BH_Drawdown': f"{bh['max_drawdown']:.2f}%",
            'BH_Sharpe': f"{bh['sharpe_ratio']:.2f}",
            'Outperformance': f"{ml['total_return'] - bh['total_return']:.2f}%"
        })
    
    # Сохраняем результаты
    results_df = pd.DataFrame(summary_data)
    results_df.to_csv('model_testing_results.csv', index=False)
    
    # Сохраняем детальные результаты
    def clean_for_json(obj):
        """Очистка объекта от циклических ссылок и numpy типов"""
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
        elif hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        else:
            return obj
    
    with open('model_testing_detailed_results.json', 'w', encoding='utf-8') as f:
        clean_results = clean_for_json(all_results)
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # Выводим сводку
    logger.info("\n📋 СВОДКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ:")
    logger.info("=" * 50)
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
    
    # Лучшие результаты
    best_return_idx = np.argmax(ml_returns)
    best_sharpe_idx = np.argmax(ml_sharpe)
    best_winrate_idx = np.argmax(ml_win_rates)
    
    logger.info(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
    logger.info(f"  Лучшая доходность: {summary_data[best_return_idx]['Symbol']} ({summary_data[best_return_idx]['ML_Return']})")
    logger.info(f"  Лучший Sharpe: {summary_data[best_sharpe_idx]['Symbol']} ({summary_data[best_sharpe_idx]['ML_Sharpe']})")
    logger.info(f"  Лучший Win Rate: {summary_data[best_winrate_idx]['Symbol']} ({summary_data[best_winrate_idx]['ML_Win_Rate']})")
    
    logger.info(f"\n💾 Результаты сохранены:")
    logger.info(f"  📄 Сводка: model_testing_results.csv")
    logger.info(f"  📄 Детали: model_testing_detailed_results.json")
    logger.info(f"  📄 Логи: model_testing.log")
    
    logger.info("\n✅ ТЕСТИРОВАНИЕ МОДЕЛЕЙ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()
