#!/usr/bin/env python3
"""
Тестирование оптимизированных технических индикаторов
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
from advanced_ml_strategies import AdvancedMLStrategies, IndicatorOptimizer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndicatorTester:
    """Тестер оптимизированных индикаторов"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def load_tbank_data(self):
        """Загрузка исторических данных из T-Bank"""
        market_data = {}
        
        # Определяем символы для тестирования
        symbols = ['GAZP', 'SBER', 'PIKK', 'SGZH', 'IRAO']
        
        # Пробуем разные пути к данным
        possible_paths = [
            'data/tbank_real',
            'data/historical',
            'data/historical/tbank_real'
        ]
        
        for symbol in symbols:
            df = None
            
            # Пробуем найти файл в разных местах
            for data_dir in possible_paths:
                possible_files = [
                    f"{symbol}_1Y_tbank.csv",
                    f"{symbol}_3M_tbank.csv", 
                    f"{symbol}_tbank.csv",
                    f"{symbol}_daily.csv"
                ]
                
                for filename in possible_files:
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        try:
                            df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
                            df = df.sort_index()
                            logger.info(f"Загружены данные {symbol} из {filepath}: {len(df)} дней")
                            break
                        except Exception as e:
                            logger.error(f"Ошибка загрузки {filepath}: {e}")
                            continue
                
                if df is not None:
                    break
            
            if df is not None:
                # Берем последние 200 дней для тестирования
                if len(df) > 200:
                    df = df.iloc[-200:]
                
                if not df.empty and len(df) >= 100:
                    market_data[symbol] = df
                else:
                    logger.warning(f"Недостаточно данных для {symbol}: {len(df)} дней")
            else:
                logger.warning(f"Файл не найден для {symbol}")
        
        return market_data
    
    def test_indicator_optimization(self):
        """Тестирование оптимизации индикаторов"""
        logger.info("🔍 ТЕСТИРОВАНИЕ ОПТИМИЗАЦИИ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ")
        logger.info("=" * 70)
        
        # Загружаем данные
        market_data = self.load_tbank_data()
        
        if not market_data:
            logger.error("❌ Нет данных для тестирования")
            return
        
        logger.info(f"✅ Загружено {len(market_data)} наборов данных")
        
        # Создаем оптимизатор индикаторов
        optimizer = IndicatorOptimizer()
        
        # Результаты оптимизации
        optimization_results = {}
        
        for symbol, data in market_data.items():
            logger.info(f"\n📈 Анализ индикаторов для {symbol}...")
            
            try:
                # Оценка всех индикаторов
                scores = optimizer.evaluate_indicators(data)
                
                if not scores:
                    logger.warning(f"Не удалось оценить индикаторы для {symbol}")
                    continue
                
                # Выбор лучших индикаторов
                best_indicators = optimizer.select_best_indicators(scores, top_n=15, min_correlation=0.05)
                
                # Сохраняем результаты
                optimization_results[symbol] = {
                    'all_scores': scores,
                    'best_indicators': best_indicators,
                    'top_10_scores': dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10])
                }
                
                logger.info(f"✅ Выбрано {len(best_indicators)} лучших индикаторов для {symbol}")
                logger.info("🏆 ТОП-10 индикаторов:")
                for i, (indicator, score) in enumerate(optimization_results[symbol]['top_10_scores'].items(), 1):
                    logger.info(f"  {i}. {indicator}: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Ошибка оптимизации для {symbol}: {e}")
        
        # Сохраняем результаты оптимизации
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = f"backtesting/results/indicator_optimization_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Результаты оптимизации сохранены в {output_path}")
        
        # Анализ общих паттернов
        self._analyze_common_patterns(optimization_results)
        
        return optimization_results
    
    def _analyze_common_patterns(self, results: dict):
        """Анализ общих паттернов в результатах"""
        logger.info("\n📊 АНАЛИЗ ОБЩИХ ПАТТЕРНОВ:")
        logger.info("=" * 50)
        
        # Собираем все индикаторы и их средние оценки
        all_indicators = {}
        indicator_counts = {}
        
        for symbol, data in results.items():
            for indicator, score in data['all_scores'].items():
                if indicator not in all_indicators:
                    all_indicators[indicator] = []
                    indicator_counts[indicator] = 0
                
                all_indicators[indicator].append(score)
                indicator_counts[indicator] += 1
        
        # Рассчитываем средние оценки
        avg_scores = {}
        for indicator, scores in all_indicators.items():
            if len(scores) >= 2:  # Минимум 2 инструмента
                avg_scores[indicator] = np.mean(scores)
        
        # Сортируем по средним оценкам
        sorted_indicators = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("🏆 УНИВЕРСАЛЬНО ЛУЧШИЕ ИНДИКАТОРЫ (средняя оценка):")
        for i, (indicator, avg_score) in enumerate(sorted_indicators[:15], 1):
            count = indicator_counts[indicator]
            logger.info(f"  {i}. {indicator}: {avg_score:.4f} (в {count} инструментах)")
        
        # Анализ по категориям
        self._analyze_by_categories(sorted_indicators)
    
    def _analyze_by_categories(self, sorted_indicators: list):
        """Анализ по категориям индикаторов"""
        logger.info("\n📈 АНАЛИЗ ПО КАТЕГОРИЯМ:")
        logger.info("=" * 40)
        
        categories = {
            'Moving Averages': ['sma', 'ema'],
            'Momentum': ['rsi', 'momentum', 'stoch', 'williams'],
            'Trend': ['macd', 'adx', 'psar', 'ichimoku'],
            'Volatility': ['bb', 'volatility', 'atr'],
            'Volume': ['volume', 'obv', 'vpt'],
            'Patterns': ['pattern', 'price_change', 'high_low_ratio']
        }
        
        for category, keywords in categories.items():
            category_indicators = []
            for indicator, score in sorted_indicators:
                if any(keyword in indicator.lower() for keyword in keywords):
                    category_indicators.append((indicator, score))
            
            if category_indicators:
                logger.info(f"\n{category}:")
                for indicator, score in category_indicators[:5]:
                    logger.info(f"  • {indicator}: {score:.4f}")
    
    def test_ml_strategies_with_optimization(self):
        """Тестирование ML стратегий с оптимизированными индикаторами"""
        logger.info("\n🤖 ТЕСТИРОВАНИЕ ML СТРАТЕГИЙ С ОПТИМИЗАЦИЕЙ")
        logger.info("=" * 60)
        
        # Загружаем данные
        market_data = self.load_tbank_data()
        
        if not market_data:
            logger.error("❌ Нет данных для тестирования")
            return
        
        # Тестируем с оптимизацией и без
        test_configs = [
            {'optimize': True, 'max_indicators': 15, 'name': 'Optimized'},
            {'optimize': False, 'max_indicators': 15, 'name': 'Standard'}
        ]
        
        all_results = {}
        
        for config in test_configs:
            logger.info(f"\n🔧 Конфигурация: {config['name']}")
            
            # Создаем стратегию
            strategy = AdvancedMLStrategies(
                initial_capital=self.initial_capital,
                optimize_indicators=config['optimize'],
                max_indicators=config['max_indicators']
            )
            
            config_results = {}
            
            for symbol, data in market_data.items():
                logger.info(f"  📊 Тестирование {symbol}...")
                
                try:
                    # Тестируем ARIMA стратегию
                    arima_result = strategy.arima_strategy(symbol, data.copy())
                    if arima_result:
                        config_results[f"{symbol}_ARIMA"] = arima_result
                        logger.info(f"    ✅ ARIMA: {arima_result['monthly_return']:.2f}% в месяц")
                    
                    # Тестируем LSTM стратегию
                    lstm_result = strategy.lstm_strategy(symbol, data.copy())
                    if lstm_result:
                        config_results[f"{symbol}_LSTM"] = lstm_result
                        logger.info(f"    ✅ LSTM: {lstm_result['monthly_return']:.2f}% в месяц")
                    
                except Exception as e:
                    logger.error(f"    ❌ Ошибка для {symbol}: {e}")
            
            all_results[config['name']] = config_results
        
        # Сравниваем результаты
        self._compare_optimization_results(all_results)
        
        return all_results
    
    def _compare_optimization_results(self, results: dict):
        """Сравнение результатов оптимизации"""
        logger.info("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ:")
        logger.info("=" * 50)
        
        if 'Optimized' not in results or 'Standard' not in results:
            logger.warning("Недостаточно данных для сравнения")
            return
        
        optimized = results['Optimized']
        standard = results['Standard']
        
        # Находим общие стратегии
        common_strategies = set(optimized.keys()) & set(standard.keys())
        
        if not common_strategies:
            logger.warning("Нет общих стратегий для сравнения")
            return
        
        logger.info("📈 СРАВНЕНИЕ МЕСЯЧНОЙ ДОХОДНОСТИ:")
        logger.info("Стратегия | Оптимизированная | Стандартная | Разница")
        logger.info("-" * 60)
        
        improvements = []
        for strategy in sorted(common_strategies):
            opt_return = optimized[strategy]['monthly_return']
            std_return = standard[strategy]['monthly_return']
            difference = opt_return - std_return
            
            improvements.append(difference)
            
            logger.info(f"{strategy:15} | {opt_return:8.2f}% | {std_return:7.2f}% | {difference:+6.2f}%")
        
        # Статистика улучшений
        avg_improvement = np.mean(improvements)
        positive_improvements = sum(1 for x in improvements if x > 0)
        total_strategies = len(improvements)
        
        logger.info(f"\n📊 СТАТИСТИКА УЛУЧШЕНИЙ:")
        logger.info(f"  Среднее улучшение: {avg_improvement:+.2f}%")
        logger.info(f"  Улучшенных стратегий: {positive_improvements}/{total_strategies}")
        logger.info(f"  Процент улучшений: {positive_improvements/total_strategies*100:.1f}%")

def main():
    """Основная функция"""
    tester = IndicatorTester()
    
    # Тестируем оптимизацию индикаторов
    optimization_results = tester.test_indicator_optimization()
    
    # Тестируем ML стратегии с оптимизацией
    ml_results = tester.test_ml_strategies_with_optimization()
    
    logger.info("\n✅ ТЕСТИРОВАНИЕ ОПТИМИЗАЦИИ ИНДИКАТОРОВ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()
