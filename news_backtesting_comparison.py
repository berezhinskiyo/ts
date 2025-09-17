#!/usr/bin/env python3
"""
Сравнительное тестирование стратегий с анализом новостей и без
Загрузка исторических данных и сравнение результатов
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Импортируем наши модули
from russian_news_analyzer import RussianNewsAnalyzer
from russian_trading_integration import RussianTradingStrategy

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsBacktestingComparison:
    """Сравнительное тестирование стратегий с новостями и без"""
    
    def __init__(self, symbols: List[str], config_file: str = "russian_news_config.json"):
        self.symbols = symbols
        self.news_analyzer = RussianNewsAnalyzer(config_file)
        self.russian_strategy = RussianTradingStrategy(symbols, config_file)
        
        # Результаты тестирования
        self.results_without_news = {}
        self.results_with_news = {}
        self.comparison_results = {}
        
        logger.info(f"✅ Инициализирован сравнительный тестер для {len(symbols)} символов")
    
    def load_historical_data(self, data_dir: str = "data/tbank_real") -> Dict[str, pd.DataFrame]:
        """Загрузка исторических данных"""
        historical_data = {}
        
        for symbol in self.symbols:
            # Ищем файлы данных для символа
            possible_files = [
                f"{data_dir}/{symbol}_1Y_tbank.csv",
                f"{data_dir}/{symbol}_3M_tbank.csv", 
                f"{data_dir}/{symbol}_1M_tbank.csv",
                f"data/historical/{symbol}_tbank.csv"
            ]
            
            data_file = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    data_file = file_path
                    break
            
            if data_file:
                try:
                    df = pd.read_csv(data_file)
                    
                    # Стандартизируем колонки
                    if 'begin' in df.columns:
                        df['begin'] = pd.to_datetime(df['begin'])
                    elif 'date' in df.columns:
                        df['begin'] = pd.to_datetime(df['date'])
                        df = df.rename(columns={'date': 'begin'})
                    
                    # Убеждаемся, что есть необходимые колонки
                    required_columns = ['begin', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_columns):
                        historical_data[symbol] = df.sort_values('begin').reset_index(drop=True)
                        logger.info(f"📊 Загружены данные для {symbol}: {len(df)} записей")
                    else:
                        logger.warning(f"⚠️ Неполные данные для {symbol}: отсутствуют колонки {required_columns}")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка загрузки данных для {symbol}: {e}")
            else:
                logger.warning(f"⚠️ Файл данных для {symbol} не найден")
        
        return historical_data
    
    def generate_sample_historical_news(self, symbols: List[str], days_back: int = 30) -> Dict[str, List[Dict]]:
        """Генерация примеров исторических новостей для тестирования"""
        
        historical_news = {}
        
        for symbol in symbols:
            news_list = []
            
            # Генерируем новости за последние 30 дней
            for i in range(days_back):
                date = datetime.now() - timedelta(days=i)
                
                # Случайные новости с разными настроениями
                news_templates = [
                    {
                        'title': f'{symbol}: Положительные результаты квартала',
                        'content': f'Компания {symbol} показала рост прибыли на 15% в последнем квартале. Аналитики повышают прогнозы.',
                        'sentiment_score': 0.7,
                        'confidence': 0.8
                    },
                    {
                        'title': f'{symbol}: Стабильное развитие бизнеса',
                        'content': f'Акции {symbol} демонстрируют устойчивую динамику. Инвесторы проявляют интерес.',
                        'sentiment_score': 0.5,
                        'confidence': 0.6
                    },
                    {
                        'title': f'{symbol}: Негативные макроэкономические факторы',
                        'content': f'На фоне ухудшения макроэкономической ситуации акции {symbol} могут снизиться.',
                        'sentiment_score': -0.6,
                        'confidence': 0.7
                    },
                    {
                        'title': f'{symbol}: Нейтральные торговые сессии',
                        'content': f'Торги {symbol} прошли без значительных изменений. Объемы торгов в норме.',
                        'sentiment_score': 0.1,
                        'confidence': 0.4
                    }
                ]
                
                # Выбираем случайную новость
                news_template = np.random.choice(news_templates)
                
                news_list.append({
                    'title': news_template['title'],
                    'content': news_template['content'],
                    'published_at': date,
                    'source': 'Test News',
                    'url': f'https://test.com/{symbol}/{i}',
                    'symbol': symbol,
                    'sentiment_score': news_template['sentiment_score'],
                    'confidence': news_template['confidence']
                })
            
            historical_news[symbol] = news_list
        
        return historical_news
    
    def backtest_strategy_without_news(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Бэктестирование стратегии без учета новостей"""
        
        logger.info("🔄 Бэктестирование стратегии БЕЗ новостей...")
        results = {}
        
        for symbol in self.symbols:
            if symbol not in historical_data:
                continue
            
            df = historical_data[symbol]
            trades = []
            equity_history = []
            capital = 100000
            position = 0
            
            # Простая стратегия на основе технических индикаторов
            for i in range(20, len(df)):  # Начинаем с 20-го элемента
                current_data = df.iloc[:i+1]
                current_price = df['close'].iloc[i]
                current_time = df['begin'].iloc[i]
                
                # Простые технические сигналы
                signal = self.generate_simple_technical_signal(current_data)
                
                # Выполняем торговлю
                if signal['action'] == 'buy' and position == 0 and signal['confidence'] > 0.5:
                    position = capital / current_price
                    capital = 0
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'time': current_time,
                        'confidence': signal['confidence']
                    })
                elif signal['action'] == 'sell' and position > 0 and signal['confidence'] > 0.5:
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
            
            results[symbol] = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'trades': trades,
                'equity_history': equity_history,
                'final_equity': final_equity,
                'strategy_type': 'technical_only'
            }
            
            logger.info(f"✅ {symbol} (без новостей): Доходность={total_return:.2f}%, "
                       f"Просадка={max_drawdown:.2f}%, Сделок={len(trades)}")
        
        return results
    
    async def backtest_strategy_with_news(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Бэктестирование стратегии с учетом новостей"""
        
        logger.info("🔄 Бэктестирование стратегии С новостями...")
        
        # Генерируем исторические новости
        historical_news = self.generate_sample_historical_news(self.symbols, days_back=30)
        
        results = {}
        
        for symbol in self.symbols:
            if symbol not in historical_data:
                continue
            
            df = historical_data[symbol]
            news = historical_news.get(symbol, [])
            trades = []
            equity_history = []
            capital = 100000
            position = 0
            
            for i in range(20, len(df)):
                current_data = df.iloc[:i+1]
                current_price = df['close'].iloc[i]
                current_time = df['begin'].iloc[i]
                
                # Фильтруем новости для текущего момента
                relevant_news = [
                    n for n in news 
                    if abs((current_time - n['published_at']).total_seconds()) < 24 * 3600
                ]
                
                # Анализируем настроения
                if relevant_news:
                    sentiment = self.calculate_news_sentiment(relevant_news)
                else:
                    sentiment = {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0}
                
                # Генерируем технические сигналы
                technical_signal = self.generate_simple_technical_signal(current_data)
                
                # Комбинируем сигналы
                combined_signal = self.combine_signals_with_news(technical_signal, sentiment)
                
                # Выполняем торговлю
                if combined_signal['action'] == 'buy' and position == 0 and combined_signal['confidence'] > 0.4:
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
                elif combined_signal['action'] == 'sell' and position > 0 and combined_signal['confidence'] > 0.4:
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
            
            results[symbol] = {
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
                'strategy_type': 'technical_with_news'
            }
            
            logger.info(f"✅ {symbol} (с новостями): Доходность={total_return:.2f}%, "
                       f"Просадка={max_drawdown:.2f}%, Сделок={len(trades)}, "
                       f"Средний сентимент={results[symbol]['avg_sentiment']:.3f}")
        
        return results
    
    def generate_simple_technical_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация простого технического сигнала"""
        
        if len(df) < 20:
            return {'action': 'hold', 'confidence': 0.0}
        
        # Простые технические индикаторы
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # RSI (упрощенный)
        price_changes = df['close'].diff().dropna()
        if len(price_changes) >= 14:
            gains = price_changes.where(price_changes > 0, 0).rolling(14).mean().iloc[-1]
            losses = (-price_changes.where(price_changes < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + gains / losses)) if losses > 0 else 50
        else:
            rsi = 50
        
        # Генерируем сигнал
        signal = 0.0
        
        # Сигнал по скользящей средней
        if current_price > sma_20 * 1.02:
            signal += 0.3
        elif current_price < sma_20 * 0.98:
            signal -= 0.3
        
        # Сигнал по RSI
        if rsi > 70:
            signal -= 0.2
        elif rsi < 30:
            signal += 0.2
        
        # Определяем действие
        if signal > 0.3:
            action = 'buy'
            confidence = min(signal, 1.0)
        elif signal < -0.3:
            action = 'sell'
            confidence = min(abs(signal), 1.0)
        else:
            action = 'hold'
            confidence = 0.0
        
        return {
            'action': action,
            'confidence': confidence,
            'signal': signal
        }
    
    def calculate_news_sentiment(self, news_list: List[Dict]) -> Dict[str, float]:
        """Расчет настроений новостей"""
        if not news_list:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0}
        
        total_sentiment = 0.0
        total_confidence = 0.0
        weight_sum = 0.0
        
        for news in news_list:
            # Взвешиваем по времени
            hours_ago = (datetime.now() - news['published_at']).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - hours_ago / (24 * 7))
            
            # Взвешиваем по уверенности
            confidence_weight = news.get('confidence', 0.5)
            
            # Общий вес
            weight = time_weight * confidence_weight
            
            total_sentiment += news['sentiment_score'] * weight
            total_confidence += confidence_weight * weight
            weight_sum += weight
        
        avg_sentiment = total_sentiment / weight_sum if weight_sum > 0 else 0.0
        avg_confidence = total_confidence / weight_sum if weight_sum > 0 else 0.0
        
        return {
            'sentiment_score': avg_sentiment,
            'confidence': avg_confidence,
            'news_count': len(news_list)
        }
    
    def combine_signals_with_news(self, technical_signal: Dict, sentiment: Dict) -> Dict[str, Any]:
        """Комбинирование технических сигналов с новостями"""
        
        # Веса для комбинирования
        technical_weight = 0.6
        sentiment_weight = 0.4
        
        # Нормализуем сигналы
        tech_signal = technical_signal.get('signal', 0.0)
        sent_signal = sentiment['sentiment_score']
        
        # Комбинируем
        combined_signal = tech_signal * technical_weight + sent_signal * sentiment_weight
        
        # Определяем действие
        if combined_signal > 0.2:
            action = 'buy'
            confidence = min(combined_signal, 1.0)
        elif combined_signal < -0.2:
            action = 'sell'
            confidence = min(abs(combined_signal), 1.0)
        else:
            action = 'hold'
            confidence = 0.0
        
        # Корректируем уверенность на основе качества новостей
        news_quality_factor = min(sentiment['news_count'] / 3.0, 1.0)
        final_confidence = confidence * (0.7 + 0.3 * news_quality_factor)
        
        return {
            'action': action,
            'confidence': final_confidence,
            'combined_signal': combined_signal
        }
    
    def compare_results(self) -> Dict[str, Any]:
        """Сравнение результатов тестирования"""
        
        logger.info("📊 Сравнение результатов тестирования...")
        
        comparison = {}
        
        for symbol in self.symbols:
            if symbol in self.results_without_news and symbol in self.results_with_news:
                without_news = self.results_without_news[symbol]
                with_news = self.results_with_news[symbol]
                
                # Рассчитываем улучшения
                return_improvement = with_news['total_return'] - without_news['total_return']
                drawdown_improvement = without_news['max_drawdown'] - with_news['max_drawdown']  # Меньше просадка = лучше
                trades_improvement = with_news['total_trades'] - without_news['total_trades']
                
                comparison[symbol] = {
                    'without_news': {
                        'total_return': without_news['total_return'],
                        'max_drawdown': without_news['max_drawdown'],
                        'total_trades': without_news['total_trades']
                    },
                    'with_news': {
                        'total_return': with_news['total_return'],
                        'max_drawdown': with_news['max_drawdown'],
                        'total_trades': with_news['total_trades'],
                        'avg_sentiment': with_news.get('avg_sentiment', 0.0),
                        'avg_news_count': with_news.get('avg_news_count', 0.0)
                    },
                    'improvements': {
                        'return_improvement': return_improvement,
                        'drawdown_improvement': drawdown_improvement,
                        'trades_improvement': trades_improvement
                    }
                }
        
        return comparison
    
    def generate_comparison_report(self, comparison: Dict[str, Any]) -> str:
        """Генерация отчета сравнения"""
        
        report = []
        report.append("📊 ОТЧЕТ СРАВНЕНИЯ СТРАТЕГИЙ")
        report.append("=" * 80)
        report.append("")
        
        # Общая статистика
        total_symbols = len(comparison)
        positive_return_improvements = sum(1 for c in comparison.values() 
                                         if c['improvements']['return_improvement'] > 0)
        positive_drawdown_improvements = sum(1 for c in comparison.values() 
                                           if c['improvements']['drawdown_improvement'] > 0)
        
        report.append("📈 ОБЩАЯ СТАТИСТИКА:")
        report.append(f"  Всего символов: {total_symbols}")
        report.append(f"  Улучшение доходности: {positive_return_improvements}/{total_symbols} "
                     f"({positive_return_improvements/total_symbols*100:.1f}%)")
        report.append(f"  Улучшение просадки: {positive_drawdown_improvements}/{total_symbols} "
                     f"({positive_drawdown_improvements/total_symbols*100:.1f}%)")
        report.append("")
        
        # Детали по символам
        report.append("📊 ДЕТАЛИ ПО СИМВОЛАМ:")
        report.append("-" * 80)
        
        for symbol, data in comparison.items():
            report.append(f"\n{symbol}:")
            report.append(f"  БЕЗ новостей:")
            report.append(f"    Доходность: {data['without_news']['total_return']:.2f}%")
            report.append(f"    Просадка: {data['without_news']['max_drawdown']:.2f}%")
            report.append(f"    Сделок: {data['without_news']['total_trades']}")
            
            report.append(f"  С новостями:")
            report.append(f"    Доходность: {data['with_news']['total_return']:.2f}%")
            report.append(f"    Просадка: {data['with_news']['max_drawdown']:.2f}%")
            report.append(f"    Сделок: {data['with_news']['total_trades']}")
            report.append(f"    Средний сентимент: {data['with_news']['avg_sentiment']:.3f}")
            report.append(f"    Среднее новостей: {data['with_news']['avg_news_count']:.1f}")
            
            report.append(f"  УЛУЧШЕНИЯ:")
            report.append(f"    Доходность: {data['improvements']['return_improvement']:+.2f}%")
            report.append(f"    Просадка: {data['improvements']['drawdown_improvement']:+.2f}%")
            report.append(f"    Сделок: {data['improvements']['trades_improvement']:+d}")
        
        # Выводы
        report.append("\n" + "=" * 80)
        report.append("🎯 ВЫВОДЫ:")
        
        avg_return_improvement = np.mean([c['improvements']['return_improvement'] 
                                        for c in comparison.values()])
        avg_drawdown_improvement = np.mean([c['improvements']['drawdown_improvement'] 
                                          for c in comparison.values()])
        
        if avg_return_improvement > 0:
            report.append(f"✅ Анализ новостей улучшает доходность в среднем на {avg_return_improvement:.2f}%")
        else:
            report.append(f"❌ Анализ новостей снижает доходность в среднем на {abs(avg_return_improvement):.2f}%")
        
        if avg_drawdown_improvement > 0:
            report.append(f"✅ Анализ новостей снижает просадку в среднем на {avg_drawdown_improvement:.2f}%")
        else:
            report.append(f"❌ Анализ новостей увеличивает просадку в среднем на {abs(avg_drawdown_improvement):.2f}%")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Сохранение результатов в файл"""
        try:
            # Конвертируем datetime объекты в строки для JSON
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
    
    async def run_full_comparison(self) -> Dict[str, Any]:
        """Запуск полного сравнительного тестирования"""
        
        logger.info("🚀 ЗАПУСК ПОЛНОГО СРАВНИТЕЛЬНОГО ТЕСТИРОВАНИЯ")
        logger.info("=" * 60)
        
        # 1. Загружаем исторические данные
        logger.info("1. Загрузка исторических данных...")
        historical_data = self.load_historical_data()
        
        if not historical_data:
            logger.error("❌ Не удалось загрузить исторические данные")
            return {}
        
        # 2. Тестируем стратегию без новостей
        logger.info("\n2. Тестирование стратегии БЕЗ новостей...")
        self.results_without_news = self.backtest_strategy_without_news(historical_data)
        
        # 3. Тестируем стратегию с новостями
        logger.info("\n3. Тестирование стратегии С новостями...")
        self.results_with_news = await self.backtest_strategy_with_news(historical_data)
        
        # 4. Сравниваем результаты
        logger.info("\n4. Сравнение результатов...")
        comparison = self.compare_results()
        
        # 5. Генерируем отчет
        report = self.generate_comparison_report(comparison)
        print("\n" + report)
        
        # 6. Сохраняем результаты
        all_results = {
            'without_news': self.results_without_news,
            'with_news': self.results_with_news,
            'comparison': comparison,
            'report': report
        }
        
        self.save_results(all_results, 'news_backtesting_results.json')
        
        return all_results
    
    async def close(self):
        """Закрытие соединений"""
        await self.news_analyzer.close()

# Пример использования
async def main():
    """Основная функция сравнительного тестирования"""
    
    # Российские символы для тестирования
    symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN']
    
    # Создаем сравнительный тестер
    tester = NewsBacktestingComparison(symbols)
    
    try:
        # Запускаем полное сравнение
        results = await tester.run_full_comparison()
        
        if results:
            print("\n✅ Сравнительное тестирование завершено успешно!")
            print("📁 Результаты сохранены в news_backtesting_results.json")
        else:
            print("\n❌ Ошибка при выполнении тестирования")
    
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")
    
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())
