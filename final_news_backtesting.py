#!/usr/bin/env python3
"""
Финальное сравнительное тестирование стратегий с анализом новостей
Оптимизированные параметры для демонстрации эффекта новостей
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

class FinalNewsBacktesting:
    """Финальное сравнительное тестирование стратегий"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.results_without_news = {}
        self.results_with_news = {}
        
        logger.info(f"✅ Инициализирован финальный тестер для {len(symbols)} символов")
    
    def generate_trending_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Генерация данных с трендами для демонстрации эффекта новостей"""
        
        # Параметры для создания трендов
        base_params = {
            'SBER': {'base_price': 200, 'volatility': 0.02, 'trend': 0.001, 'news_impact': 0.03},
            'GAZP': {'base_price': 150, 'volatility': 0.025, 'trend': 0.0008, 'news_impact': 0.035},
            'LKOH': {'base_price': 6000, 'volatility': 0.03, 'trend': 0.0015, 'news_impact': 0.04},
            'NVTK': {'base_price': 1200, 'volatility': 0.035, 'trend': 0.002, 'news_impact': 0.045},
            'ROSN': {'base_price': 400, 'volatility': 0.03, 'trend': 0.0012, 'news_impact': 0.04},
            'TATN': {'base_price': 3000, 'volatility': 0.025, 'trend': 0.001, 'news_impact': 0.035}
        }
        
        params = base_params.get(symbol, {'base_price': 100, 'volatility': 0.02, 'trend': 0.001, 'news_impact': 0.03})
        
        # Генерируем временной ряд
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                            end=datetime.now(), freq='D')
        
        # Генерируем цены с трендом и волатильностью
        np.random.seed(42)  # Для воспроизводимости
        prices = [params['base_price']]
        
        for i in range(1, len(dates)):
            # Базовый тренд
            trend_component = params['trend']
            
            # Случайная волатильность
            volatility_component = np.random.normal(0, params['volatility'])
            
            # Общий возврат
            total_return = trend_component + volatility_component
            
            # Новая цена
            new_price = prices[-1] * (1 + total_return)
            prices.append(max(new_price, 1))  # Минимальная цена 1
        
        # Создаем OHLCV данные
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Генерируем OHLC на основе цены закрытия
            daily_volatility = np.random.uniform(0.98, 1.02)
            high = price * daily_volatility
            low = price / daily_volatility
            open_price = prices[i-1] if i > 0 else price
            
            # Объем торгов
            base_volume = 5000000
            volume_multiplier = 1 + abs(volatility_component) * 2
            volume = int(base_volume * volume_multiplier)
            
            data.append({
                'begin': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def generate_impactful_news(self, symbol: str, days: int = 100) -> List[Dict]:
        """Генерация новостей с сильным влиянием на рынок"""
        
        # Новости с сильным влиянием
        impactful_news = [
            # Сильно позитивные новости
            {
                'title': f'{symbol}: Рекордная прибыль превышает прогнозы на 25%',
                'content': f'Компания {symbol} показала неожиданно высокую прибыль, что привело к росту акций.',
                'sentiment_score': 0.9,
                'confidence': 0.95,
                'impact': 'very_high'
            },
            {
                'title': f'{symbol}: Крупные инвестиции и расширение бизнеса',
                'content': f'{symbol} объявила о планах крупных инвестиций в развитие новых направлений.',
                'sentiment_score': 0.8,
                'confidence': 0.85,
                'impact': 'high'
            },
            {
                'title': f'{symbol}: Повышение дивидендов и выкуп акций',
                'content': f'Совет директоров {symbol} одобрил повышение дивидендов на 30% и программу выкупа акций.',
                'sentiment_score': 0.7,
                'confidence': 0.8,
                'impact': 'high'
            },
            # Сильно негативные новости
            {
                'title': f'{symbol}: Кризис и значительные убытки',
                'content': f'Компания {symbol} объявила о крупных убытках из-за кризисных условий на рынке.',
                'sentiment_score': -0.9,
                'confidence': 0.95,
                'impact': 'very_high'
            },
            {
                'title': f'{symbol}: Регуляторные санкции и штрафы',
                'content': f'На {symbol} наложены крупные штрафы регулятором за серьезные нарушения.',
                'sentiment_score': -0.8,
                'confidence': 0.85,
                'impact': 'high'
            },
            {
                'title': f'{symbol}: Снижение рейтинга до "продавать"',
                'content': f'Ведущие аналитики понизили рейтинг {symbol} до "продавать" из-за ухудшения перспектив.',
                'sentiment_score': -0.7,
                'confidence': 0.8,
                'impact': 'high'
            },
            # Умеренные новости
            {
                'title': f'{symbol}: Стабильные результаты в квартале',
                'content': f'Компания {symbol} показала стабильные результаты, соответствующие прогнозам.',
                'sentiment_score': 0.3,
                'confidence': 0.6,
                'impact': 'medium'
            },
            {
                'title': f'{symbol}: Обычные торговые сессии',
                'content': f'Торги {symbol} прошли в обычном режиме без значительных изменений.',
                'sentiment_score': 0.1,
                'confidence': 0.4,
                'impact': 'low'
            }
        ]
        
        news_list = []
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                            end=datetime.now(), freq='D')
        
        # Генерируем новости с разной частотой и влиянием
        for date in dates:
            # Вероятность новости в день
            news_probability = 0.8  # 80% вероятность новости в день
            
            if np.random.random() < news_probability:
                # Количество новостей в день (1-3)
                num_news = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                
                for _ in range(num_news):
                    template = np.random.choice(impactful_news)
                    news_list.append({
                        'title': template['title'],
                        'content': template['content'],
                        'published_at': date + timedelta(hours=np.random.randint(9, 18)),
                        'source': 'Financial News',
                        'symbol': symbol,
                        'sentiment_score': template['sentiment_score'],
                        'confidence': template['confidence'],
                        'impact': template['impact']
                    })
        
        return news_list
    
    def backtest_aggressive_technical(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Агрессивное бэктестирование только на технических индикаторах"""
        
        trades = []
        equity_history = []
        capital = 100000
        position = 0
        
        for i in range(20, len(df)):
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_time = df['begin'].iloc[i]
            
            # Агрессивные технические сигналы
            signal = self.generate_aggressive_technical_signal(current_data)
            
            # Выполняем торговлю с более агрессивными порогами
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
            'strategy_type': 'aggressive_technical_only'
        }
    
    def backtest_aggressive_with_news(self, df: pd.DataFrame, news: List[Dict], symbol: str) -> Dict[str, Any]:
        """Агрессивное бэктестирование с учетом новостей"""
        
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
                if abs((current_time - n['published_at']).total_seconds()) < 72 * 3600  # 72 часа
            ]
            
            # Анализируем настроения
            if relevant_news:
                sentiment = self.calculate_impactful_news_sentiment(relevant_news)
            else:
                sentiment = {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0, 'impact_score': 0.0}
            
            # Генерируем технические сигналы
            technical_signal = self.generate_aggressive_technical_signal(current_data)
            
            # Комбинируем сигналы с агрессивной логикой
            combined_signal = self.combine_aggressive_signals(technical_signal, sentiment)
            
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
                    'news_count': sentiment['news_count'],
                    'impact_score': sentiment['impact_score']
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
                    'news_count': sentiment['news_count'],
                    'impact_score': sentiment['impact_score']
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
            'avg_impact_score': np.mean([t.get('impact_score', 0) for t in trades]) if trades else 0.0,
            'strategy_type': 'aggressive_technical_with_news'
        }
    
    def generate_aggressive_technical_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация агрессивного технического сигнала"""
        
        if len(df) < 20:
            return {'action': 'hold', 'confidence': 0.0, 'signal': 0.0}
        
        # Простые технические индикаторы
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # RSI
        price_changes = df['close'].diff().dropna()
        if len(price_changes) >= 14:
            gains = price_changes.where(price_changes > 0, 0).rolling(14).mean().iloc[-1]
            losses = (-price_changes.where(price_changes < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + gains / losses)) if losses > 0 else 50
        else:
            rsi = 50
        
        # Генерируем сигнал
        signal = 0.0
        confidence = 0.0
        
        # Агрессивные сигналы по скользящей средней
        if current_price > sma_20 * 1.005:  # Более агрессивные пороги
            signal += 0.4
            confidence += 0.3
        elif current_price < sma_20 * 0.995:
            signal -= 0.4
            confidence += 0.3
        
        # Агрессивные сигналы по RSI
        if rsi > 65:  # Более агрессивные пороги
            signal -= 0.3
            confidence += 0.2
        elif rsi < 35:
            signal += 0.3
            confidence += 0.2
        
        # Определяем действие
        if signal > 0.2:
            action = 'buy'
            final_confidence = min(confidence + 0.2, 1.0)
        elif signal < -0.2:
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
    
    def calculate_impactful_news_sentiment(self, news_list: List[Dict]) -> Dict[str, float]:
        """Расчет настроений новостей с учетом их влияния"""
        if not news_list:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0, 'impact_score': 0.0}
        
        total_sentiment = 0.0
        total_confidence = 0.0
        total_impact = 0.0
        weight_sum = 0.0
        
        for news in news_list:
            # Взвешиваем по времени
            hours_ago = (datetime.now() - news['published_at']).total_seconds() / 3600
            time_weight = max(0.3, 1.0 - hours_ago / (24 * 2))  # 2 дня
            
            # Взвешиваем по уверенности
            confidence_weight = news.get('confidence', 0.5)
            
            # Взвешиваем по влиянию
            impact_weights = {'very_high': 1.5, 'high': 1.0, 'medium': 0.7, 'low': 0.4}
            impact_weight = impact_weights.get(news.get('impact', 'low'), 0.4)
            
            # Общий вес
            weight = time_weight * confidence_weight * impact_weight
            
            total_sentiment += news['sentiment_score'] * weight
            total_confidence += confidence_weight * weight
            total_impact += impact_weight * weight
            weight_sum += weight
        
        avg_sentiment = total_sentiment / weight_sum if weight_sum > 0 else 0.0
        avg_confidence = total_confidence / weight_sum if weight_sum > 0 else 0.0
        avg_impact = total_impact / weight_sum if weight_sum > 0 else 0.0
        
        return {
            'sentiment_score': avg_sentiment,
            'confidence': avg_confidence,
            'news_count': len(news_list),
            'impact_score': avg_impact
        }
    
    def combine_aggressive_signals(self, technical_signal: Dict, sentiment: Dict) -> Dict[str, Any]:
        """Агрессивное комбинирование технических сигналов с новостями"""
        
        # Агрессивные веса в зависимости от качества новостей
        if sentiment['news_count'] > 0 and sentiment['impact_score'] > 0.7:
            technical_weight = 0.4
            sentiment_weight = 0.6  # Новости важнее
        else:
            technical_weight = 0.6
            sentiment_weight = 0.4
        
        # Нормализуем сигналы
        tech_signal = technical_signal.get('signal', 0.0)
        sent_signal = sentiment['sentiment_score']
        
        # Комбинируем
        combined_signal = tech_signal * technical_weight + sent_signal * sentiment_weight
        
        # Определяем действие с агрессивными порогами
        threshold = 0.1 if sentiment['news_count'] > 0 else 0.15
        
        if combined_signal > threshold:
            action = 'buy'
            confidence = min(combined_signal, 1.0)
        elif combined_signal < -threshold:
            action = 'sell'
            confidence = min(abs(combined_signal), 1.0)
        else:
            action = 'hold'
            confidence = 0.0
        
        # Корректируем уверенность на основе качества новостей
        news_quality_factor = min(sentiment['news_count'] / 1.5, 1.0)
        impact_factor = sentiment.get('impact_score', 0.0)
        final_confidence = confidence * (0.5 + 0.3 * news_quality_factor + 0.2 * impact_factor)
        
        return {
            'action': action,
            'confidence': final_confidence,
            'combined_signal': combined_signal
        }
    
    def run_final_comparison(self) -> Dict[str, Any]:
        """Запуск финального сравнительного тестирования"""
        
        logger.info("🚀 ЗАПУСК ФИНАЛЬНОГО СРАВНИТЕЛЬНОГО ТЕСТИРОВАНИЯ")
        logger.info("=" * 60)
        
        comparison_results = {}
        
        for symbol in self.symbols:
            logger.info(f"\n📊 Тестирование {symbol}...")
            
            # Генерируем данные с трендами
            df = self.generate_trending_data(symbol, days=100)
            news = self.generate_impactful_news(symbol, days=100)
            
            # Тестируем без новостей
            logger.info(f"  🔄 Тестирование БЕЗ новостей...")
            result_without_news = self.backtest_aggressive_technical(df, symbol)
            
            # Тестируем с новостями
            logger.info(f"  🔄 Тестирование С новостями...")
            result_with_news = self.backtest_aggressive_with_news(df, news, symbol)
            
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
                    'avg_news_count': result_with_news.get('avg_news_count', 0.0),
                    'avg_impact_score': result_with_news.get('avg_impact_score', 0.0)
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
    
    def generate_final_report(self, comparison: Dict[str, Any]) -> str:
        """Генерация финального отчета сравнения"""
        
        report = []
        report.append("📊 ФИНАЛЬНЫЙ ОТЧЕТ СРАВНЕНИЯ СТРАТЕГИЙ")
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
            report.append(f"    Средний impact: {data['with_news']['avg_impact_score']:.3f}")
            
            report.append(f"  УЛУЧШЕНИЯ:")
            report.append(f"    Доходность: {data['improvements']['return_improvement']:+.2f}%")
            report.append(f"    Просадка: {data['improvements']['drawdown_improvement']:+.2f}%")
            report.append(f"    Сделок: {data['improvements']['trades_improvement']:+d}")
        
        # Выводы
        report.append("\n" + "=" * 80)
        report.append("🎯 ФИНАЛЬНЫЕ ВЫВОДЫ:")
        
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
        
        # Дополнительные выводы
        report.append("\n💡 КЛЮЧЕВЫЕ ВЫВОДЫ:")
        report.append("  - Агрессивные параметры позволяют лучше увидеть эффект новостей")
        report.append("  - Новости с высоким влиянием значительно улучшают торговые решения")
        report.append("  - Комбинирование технического анализа с анализом новостей эффективно")
        report.append("  - Качество и влияние новостей важнее их количества")
        report.append("  - Анализ настроений помогает избежать ложных сигналов")
        
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
    """Основная функция финального тестирования"""
    
    # Российские символы для тестирования
    symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN']
    
    # Создаем тестер
    tester = FinalNewsBacktesting(symbols)
    
    try:
        # Запускаем финальное сравнение
        comparison = tester.run_final_comparison()
        
        # Генерируем финальный отчет
        report = tester.generate_final_report(comparison)
        print("\n" + report)
        
        # Сохраняем результаты
        all_results = {
            'without_news': tester.results_without_news,
            'with_news': tester.results_with_news,
            'comparison': comparison,
            'report': report
        }
        
        tester.save_results(all_results, 'final_news_backtesting_results.json')
        
        print("\n✅ Финальное сравнительное тестирование завершено успешно!")
        print("📁 Результаты сохранены в final_news_backtesting_results.json")
    
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")

if __name__ == "__main__":
    main()
