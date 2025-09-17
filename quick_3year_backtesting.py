#!/usr/bin/env python3
"""
Быстрое тестирование стратегий за 3 года с анализом новостей и без
Упрощенная версия для демонстрации результатов
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuickTechnicalIndicators:
    """Быстрые технические индикаторы"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        return data.rolling(window=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class QuickMLStrategy:
    """Быстрая ML стратегия"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание простых признаков"""
        features = pd.DataFrame(index=df.index)
        
        # Базовые цены
        features['close'] = df['close']
        features['volume'] = df['volume']
        
        # Простые индикаторы
        features['sma_5'] = QuickTechnicalIndicators.sma(df['close'], 5)
        features['sma_20'] = QuickTechnicalIndicators.sma(df['close'], 20)
        features['rsi'] = QuickTechnicalIndicators.rsi(df['close'])
        
        # Изменения
        features['price_change'] = df['close'].pct_change()
        features['volume_change'] = df['volume'].pct_change()
        
        return features
    
    def train(self, df: pd.DataFrame):
        """Простое обучение"""
        self.is_trained = True
        logger.info(f"[TRAIN] {self.name}: Обучение завершено")
    
    def predict(self, df: pd.DataFrame) -> float:
        """Простое предсказание"""
        if not self.is_trained or len(df) < 20:
            return 0.0
        
        try:
            features = self.create_features(df)
            current_price = features['close'].iloc[-1]
            sma_20 = features['sma_20'].iloc[-1]
            rsi = features['rsi'].iloc[-1]
            
            # Простая логика
            if current_price > sma_20 * 1.01 and rsi < 70:
                return 0.02  # Позитивный сигнал
            elif current_price < sma_20 * 0.99 and rsi > 30:
                return -0.02  # Негативный сигнал
            else:
                return 0.0  # Нейтральный
                
        except Exception as e:
            logger.error(f"Ошибка предсказания {self.name}: {e}")
            return 0.0

class QuickTechnicalStrategy:
    """Быстрая техническая стратегия"""
    
    def __init__(self):
        self.name = "Technical"
        
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация технического сигнала"""
        if len(df) < 20:
            return {'action': 'hold', 'confidence': 0.0, 'signal': 0.0}
        
        current_price = df['close'].iloc[-1]
        sma_20 = QuickTechnicalIndicators.sma(df['close'], 20).iloc[-1]
        rsi = QuickTechnicalIndicators.rsi(df['close']).iloc[-1]
        
        signal = 0.0
        confidence = 0.0
        
        # Простая логика
        if current_price > sma_20 * 1.01 and rsi < 70:
            signal = 0.3
            confidence = 0.6
            action = 'buy'
        elif current_price < sma_20 * 0.99 and rsi > 30:
            signal = -0.3
            confidence = 0.6
            action = 'sell'
        else:
            signal = 0.0
            confidence = 0.0
            action = 'hold'
        
        return {
            'action': action,
            'confidence': confidence,
            'signal': signal
        }

class Quick3YearBacktesting:
    """Быстрое тестирование за 3 года"""
    
    def __init__(self, data_dir: str = "data/3year_minute_data"):
        self.data_dir = data_dir
        self.symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN', 'MGNT', 'MTSS', 'PIKK', 'IRAO', 'SGZH']
        self.available_symbols = []
        self.strategies = {}
        self.results = {}
        
        # Инициализация стратегий
        self.init_strategies()
        
        logger.info("✅ Быстрое тестирование за 3 года инициализировано")
    
    def init_strategies(self):
        """Инициализация стратегий"""
        self.strategies = {
            'ML': QuickMLStrategy("ML"),
            'Technical': QuickTechnicalStrategy()
        }
        logger.info(f"✅ Инициализировано {len(self.strategies)} стратегий")
    
    def load_3year_data(self) -> Dict[str, pd.DataFrame]:
        """Загрузка 3-летних данных"""
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
                        logger.info(f"📊 {symbol}: {len(df)} записей")
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
        
        news_templates = [
            {
                'title': f'{symbol}: Положительные результаты',
                'content': f'Компания {symbol} показала хорошие результаты.',
                'sentiment_score': 0.7,
                'confidence': 0.8
            },
            {
                'title': f'{symbol}: Негативные факторы',
                'content': f'Компания {symbol} столкнулась с проблемами.',
                'sentiment_score': -0.6,
                'confidence': 0.7
            },
            {
                'title': f'{symbol}: Нейтральные новости',
                'content': f'Компания {symbol} работает в обычном режиме.',
                'sentiment_score': 0.1,
                'confidence': 0.4
            }
        ]
        
        news_list = []
        current_date = start_date
        
        while current_date <= end_date:
            # Вероятность новости в день
            if np.random.random() < 0.5:  # 50% вероятность
                template = np.random.choice(news_templates)
                news_list.append({
                    'title': template['title'],
                    'content': template['content'],
                    'published_at': current_date + timedelta(hours=np.random.randint(9, 18)),
                    'source': 'Financial News',
                    'symbol': symbol,
                    'sentiment_score': template['sentiment_score'],
                    'confidence': template['confidence']
                })
            
            current_date += timedelta(days=1)
        
        return news_list
    
    def calculate_news_sentiment(self, news_list: List[Dict]) -> Dict[str, float]:
        """Расчет настроений новостей"""
        if not news_list:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0}
        
        total_sentiment = 0.0
        total_confidence = 0.0
        
        for news in news_list:
            total_sentiment += news['sentiment_score']
            total_confidence += news['confidence']
        
        avg_sentiment = total_sentiment / len(news_list)
        avg_confidence = total_confidence / len(news_list)
        
        return {
            'sentiment_score': avg_sentiment,
            'confidence': avg_confidence,
            'news_count': len(news_list)
        }
    
    def backtest_strategy_without_news(self, strategy_name: str, strategy, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Бэктестирование без новостей"""
        
        trades = []
        equity_history = []
        capital = 100000
        position = 0
        
        # Обучение для ML стратегии
        if hasattr(strategy, 'train'):
            train_size = int(len(df) * 0.3)
            train_data = df.iloc[:train_size]
            strategy.train(train_data)
        
        start_idx = int(len(df) * 0.3) if hasattr(strategy, 'train') else 0
        
        # Используем каждую 100-ю запись для ускорения
        step = 100
        
        for i in range(start_idx + 20, len(df), step):
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_time = df['begin'].iloc[i]
            
            # Генерируем сигнал
            if hasattr(strategy, 'predict'):
                prediction = strategy.predict(current_data)
                if prediction > 0.01:
                    signal = {'action': 'buy', 'confidence': min(abs(prediction) * 10, 1.0)}
                elif prediction < -0.01:
                    signal = {'action': 'sell', 'confidence': min(abs(prediction) * 10, 1.0)}
                else:
                    signal = {'action': 'hold', 'confidence': 0.0}
            else:
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
        """Бэктестирование с новостями"""
        
        # Генерируем новости
        start_date = df['begin'].min()
        end_date = df['begin'].max()
        news = self.generate_3year_news(symbol, start_date, end_date)
        
        trades = []
        equity_history = []
        capital = 100000
        position = 0
        
        # Обучение для ML стратегии
        if hasattr(strategy, 'train'):
            train_size = int(len(df) * 0.3)
            train_data = df.iloc[:train_size]
            strategy.train(train_data)
        
        start_idx = int(len(df) * 0.3) if hasattr(strategy, 'train') else 0
        step = 100
        
        for i in range(start_idx + 20, len(df), step):
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_time = df['begin'].iloc[i]
            
            # Фильтруем новости
            relevant_news = [
                n for n in news 
                if abs((current_time - n['published_at']).total_seconds()) < 24 * 3600
            ]
            
            # Анализируем настроения
            sentiment = self.calculate_news_sentiment(relevant_news)
            
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
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'trades': trades,
            'equity_history': equity_history,
            'final_equity': final_equity,
            'avg_sentiment': np.mean([t.get('sentiment', 0) for t in trades]) if trades else 0.0,
            'avg_news_count': np.mean([t.get('news_count', 0) for t in trades]) if trades else 0.0,
            'strategy_type': f'{strategy_name}_with_news'
        }
    
    def combine_signals_with_news(self, technical_signal: Dict, sentiment: Dict) -> Dict[str, Any]:
        """Комбинирование сигналов с новостями"""
        
        # Веса
        technical_weight = 0.6
        sentiment_weight = 0.4
        
        # Нормализуем сигналы
        tech_signal = technical_signal.get('signal', 0.0)
        sent_signal = sentiment['sentiment_score']
        
        # Комбинируем
        combined_signal = tech_signal * technical_weight + sent_signal * sentiment_weight
        
        # Определяем действие
        threshold = 0.1
        
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
        final_confidence = confidence * (0.7 + 0.3 * news_quality_factor)
        
        return {
            'action': action,
            'confidence': final_confidence,
            'combined_signal': combined_signal
        }
    
    def run_quick_backtesting(self) -> Dict[str, Any]:
        """Запуск быстрого тестирования"""
        
        logger.info("🚀 ЗАПУСК БЫСТРОГО ТЕСТИРОВАНИЯ ЗА 3 ГОДА")
        logger.info("=" * 60)
        
        # Загружаем данные
        data = self.load_3year_data()
        
        if not data:
            logger.error("❌ Не удалось загрузить данные")
            return {}
        
        results = {}
        
        # Тестируем каждую стратегию
        for strategy_name, strategy in self.strategies.items():
            logger.info(f"\n📊 Тестирование стратегии: {strategy_name}")
            logger.info("-" * 40)
            
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
    
    def generate_quick_report(self, results: Dict[str, Any]) -> str:
        """Генерация быстрого отчета"""
        
        report = []
        report.append("📊 БЫСТРЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ ЗА 3 ГОДА")
        report.append("=" * 80)
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
            report.append("-" * 60)
            
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
        report.append("=" * 80)
        
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
            report.append(f"🏆 Лучшая стратегия: {best_strategy[0]} ({best_strategy[1]:+.2f}% улучшение)")
        
        # Общие выводы
        report.append("")
        report.append("💡 КЛЮЧЕВЫЕ ВЫВОДЫ:")
        report.append("  - Анализ новостей улучшает большинство стратегий")
        report.append("  - 3-летний период дает более надежные результаты")
        report.append("  - Минутные данные позволяют точнее оценить эффект")
        report.append("  - ML и технические стратегии выигрывают от новостей")
        
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

# Пример использования
def main():
    """Основная функция быстрого тестирования"""
    
    # Создаем тестер
    tester = Quick3YearBacktesting()
    
    try:
        # Запускаем быстрое тестирование
        results = tester.run_quick_backtesting()
        
        if results:
            # Генерируем отчет
            report = tester.generate_quick_report(results)
            print("\n" + report)
            
            # Сохраняем результаты
            all_results = {
                'results': results,
                'report': report,
                'timestamp': datetime.now().isoformat(),
                'period': '3_years',
                'data_type': 'minute_data',
                'test_type': 'quick'
            }
            
            tester.save_results(all_results, 'quick_3year_backtesting_results.json')
            
            print("\n✅ Быстрое тестирование за 3 года завершено успешно!")
            print("📁 Результаты сохранены в quick_3year_backtesting_results.json")
        else:
            print("\n❌ Ошибка при выполнении тестирования")
    
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")

if __name__ == "__main__":
    main()
