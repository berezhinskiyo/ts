#!/usr/bin/env python3
"""
Упрощенное сравнительное тестирование стратегий с анализом новостей и без
Использует синтетические данные для демонстрации
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleNewsBacktesting:
    """Упрощенное сравнительное тестирование стратегий"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.results_without_news = {}
        self.results_with_news = {}
        
        logger.info(f"✅ Инициализирован упрощенный тестер для {len(symbols)} символов")
    
    def generate_synthetic_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Генерация синтетических данных для тестирования"""
        
        # Базовые параметры для разных символов
        base_params = {
            'SBER': {'base_price': 200, 'volatility': 0.02, 'trend': 0.001},
            'GAZP': {'base_price': 150, 'volatility': 0.025, 'trend': 0.0005},
            'LKOH': {'base_price': 6000, 'volatility': 0.03, 'trend': 0.002},
            'NVTK': {'base_price': 1200, 'volatility': 0.035, 'trend': 0.0015},
            'ROSN': {'base_price': 400, 'volatility': 0.03, 'trend': 0.001},
            'TATN': {'base_price': 3000, 'volatility': 0.025, 'trend': 0.0008}
        }
        
        params = base_params.get(symbol, {'base_price': 100, 'volatility': 0.02, 'trend': 0.001})
        
        # Генерируем временной ряд
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                            end=datetime.now(), freq='D')
        
        # Генерируем цены с трендом и волатильностью
        np.random.seed(42)  # Для воспроизводимости
        returns = np.random.normal(params['trend'], params['volatility'], len(dates))
        prices = [params['base_price']]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1))  # Минимальная цена 1
        
        # Создаем OHLCV данные
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Генерируем OHLC на основе цены закрытия
            volatility_factor = np.random.uniform(0.95, 1.05)
            high = price * volatility_factor
            low = price / volatility_factor
            open_price = prices[i-1] if i > 0 else price
            
            # Объем торгов
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'begin': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def generate_synthetic_news(self, symbol: str, days: int = 100) -> List[Dict]:
        """Генерация синтетических новостей"""
        
        news_templates = [
            {
                'title': f'{symbol}: Положительные результаты квартала',
                'content': f'Компания {symbol} показала рост прибыли. Аналитики повышают прогнозы.',
                'sentiment_score': 0.7,
                'confidence': 0.8
            },
            {
                'title': f'{symbol}: Стабильное развитие бизнеса',
                'content': f'Акции {symbol} демонстрируют устойчивую динамику.',
                'sentiment_score': 0.5,
                'confidence': 0.6
            },
            {
                'title': f'{symbol}: Негативные макроэкономические факторы',
                'content': f'На фоне ухудшения ситуации акции {symbol} могут снизиться.',
                'sentiment_score': -0.6,
                'confidence': 0.7
            },
            {
                'title': f'{symbol}: Нейтральные торговые сессии',
                'content': f'Торги {symbol} прошли без значительных изменений.',
                'sentiment_score': 0.1,
                'confidence': 0.4
            }
        ]
        
        news_list = []
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                            end=datetime.now(), freq='D')
        
        # Генерируем 1-3 новости в день
        for date in dates:
            num_news = np.random.randint(1, 4)
            for _ in range(num_news):
                template = np.random.choice(news_templates)
                news_list.append({
                    'title': template['title'],
                    'content': template['content'],
                    'published_at': date + timedelta(hours=np.random.randint(0, 24)),
                    'source': 'Synthetic News',
                    'symbol': symbol,
                    'sentiment_score': template['sentiment_score'],
                    'confidence': template['confidence']
                })
        
        return news_list
    
    def backtest_technical_only(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Бэктестирование только на технических индикаторах"""
        
        trades = []
        equity_history = []
        capital = 100000
        position = 0
        
        for i in range(20, len(df)):
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_time = df['begin'].iloc[i]
            
            # Простые технические сигналы
            signal = self.generate_technical_signal(current_data)
            
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
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'trades': trades,
            'equity_history': equity_history,
            'final_equity': final_equity,
            'strategy_type': 'technical_only'
        }
    
    def backtest_with_news(self, df: pd.DataFrame, news: List[Dict], symbol: str) -> Dict[str, Any]:
        """Бэктестирование с учетом новостей"""
        
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
            technical_signal = self.generate_technical_signal(current_data)
            
            # Комбинируем сигналы
            combined_signal = self.combine_signals(technical_signal, sentiment)
            
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
            'strategy_type': 'technical_with_news'
        }
    
    def generate_technical_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация технического сигнала"""
        
        if len(df) < 20:
            return {'action': 'hold', 'confidence': 0.0, 'signal': 0.0}
        
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
    
    def combine_signals(self, technical_signal: Dict, sentiment: Dict) -> Dict[str, Any]:
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
    
    def run_comparison(self) -> Dict[str, Any]:
        """Запуск сравнительного тестирования"""
        
        logger.info("🚀 ЗАПУСК УПРОЩЕННОГО СРАВНИТЕЛЬНОГО ТЕСТИРОВАНИЯ")
        logger.info("=" * 60)
        
        comparison_results = {}
        
        for symbol in self.symbols:
            logger.info(f"\n📊 Тестирование {symbol}...")
            
            # Генерируем синтетические данные
            df = self.generate_synthetic_data(symbol, days=100)
            news = self.generate_synthetic_news(symbol, days=100)
            
            # Тестируем без новостей
            logger.info(f"  🔄 Тестирование БЕЗ новостей...")
            result_without_news = self.backtest_technical_only(df, symbol)
            
            # Тестируем с новостями
            logger.info(f"  🔄 Тестирование С новостями...")
            result_with_news = self.backtest_with_news(df, news, symbol)
            
            # Сохраняем результаты
            self.results_without_news[symbol] = result_without_news
            self.results_with_news[symbol] = result_with_news
            
            # Рассчитываем улучшения
            return_improvement = result_with_news['total_return'] - result_without_news['total_return']
            drawdown_improvement = result_without_news['max_drawdown'] - result_with_news['max_drawdown']
            trades_improvement = result_with_news['total_trades'] - result_without_news['total_trades']
            
            comparison_results[symbol] = {
                'without_news': {
                    'total_return': result_without_news['total_return'],
                    'max_drawdown': result_without_news['max_drawdown'],
                    'total_trades': result_without_news['total_trades']
                },
                'with_news': {
                    'total_return': result_with_news['total_return'],
                    'max_drawdown': result_with_news['max_drawdown'],
                    'total_trades': result_with_news['total_trades'],
                    'avg_sentiment': result_with_news.get('avg_sentiment', 0.0),
                    'avg_news_count': result_with_news.get('avg_news_count', 0.0)
                },
                'improvements': {
                    'return_improvement': return_improvement,
                    'drawdown_improvement': drawdown_improvement,
                    'trades_improvement': trades_improvement
                }
            }
            
            logger.info(f"  ✅ {symbol}: Без новостей={result_without_news['total_return']:.2f}%, "
                       f"С новостями={result_with_news['total_return']:.2f}%, "
                       f"Улучшение={return_improvement:+.2f}%")
        
        return comparison_results
    
    def generate_report(self, comparison: Dict[str, Any]) -> str:
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

# Пример использования
def main():
    """Основная функция упрощенного тестирования"""
    
    # Российские символы для тестирования
    symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN']
    
    # Создаем тестер
    tester = SimpleNewsBacktesting(symbols)
    
    try:
        # Запускаем сравнение
        comparison = tester.run_comparison()
        
        # Генерируем отчет
        report = tester.generate_report(comparison)
        print("\n" + report)
        
        # Сохраняем результаты
        all_results = {
            'without_news': tester.results_without_news,
            'with_news': tester.results_with_news,
            'comparison': comparison,
            'report': report
        }
        
        tester.save_results(all_results, 'simple_news_backtesting_results.json')
        
        print("\n✅ Упрощенное сравнительное тестирование завершено успешно!")
        print("📁 Результаты сохранены в simple_news_backtesting_results.json")
    
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")

if __name__ == "__main__":
    main()
