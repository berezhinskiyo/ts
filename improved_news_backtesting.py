#!/usr/bin/env python3
"""
Улучшенное сравнительное тестирование стратегий с анализом новостей
Более реалистичные параметры и лучшая логика торговли
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

class ImprovedNewsBacktesting:
    """Улучшенное сравнительное тестирование стратегий"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.results_without_news = {}
        self.results_with_news = {}
        
        logger.info(f"✅ Инициализирован улучшенный тестер для {len(symbols)} символов")
    
    def generate_realistic_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Генерация более реалистичных данных для тестирования"""
        
        # Базовые параметры для разных символов
        base_params = {
            'SBER': {'base_price': 200, 'volatility': 0.015, 'trend': 0.0008, 'mean_reversion': 0.1},
            'GAZP': {'base_price': 150, 'volatility': 0.018, 'trend': 0.0005, 'mean_reversion': 0.08},
            'LKOH': {'base_price': 6000, 'volatility': 0.022, 'trend': 0.0012, 'mean_reversion': 0.12},
            'NVTK': {'base_price': 1200, 'volatility': 0.025, 'trend': 0.0015, 'mean_reversion': 0.15},
            'ROSN': {'base_price': 400, 'volatility': 0.020, 'trend': 0.0010, 'mean_reversion': 0.10},
            'TATN': {'base_price': 3000, 'volatility': 0.018, 'trend': 0.0008, 'mean_reversion': 0.09}
        }
        
        params = base_params.get(symbol, {'base_price': 100, 'volatility': 0.02, 'trend': 0.001, 'mean_reversion': 0.1})
        
        # Генерируем временной ряд
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                            end=datetime.now(), freq='D')
        
        # Генерируем более реалистичные цены с трендом, волатильностью и mean reversion
        np.random.seed(42)  # Для воспроизводимости
        prices = [params['base_price']]
        
        for i in range(1, len(dates)):
            # Базовый тренд
            trend_component = params['trend']
            
            # Случайная волатильность
            volatility_component = np.random.normal(0, params['volatility'])
            
            # Mean reversion (возврат к среднему)
            mean_reversion_component = -params['mean_reversion'] * (prices[-1] - params['base_price']) / params['base_price']
            
            # Общий возврат
            total_return = trend_component + volatility_component + mean_reversion_component
            
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
            
            # Объем торгов (зависит от волатильности)
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
    
    def generate_realistic_news(self, symbol: str, days: int = 100) -> List[Dict]:
        """Генерация более реалистичных новостей"""
        
        # Различные типы новостей с разными влияниями
        news_templates = [
            # Позитивные новости
            {
                'title': f'{symbol}: Рекордная прибыль в квартале',
                'content': f'Компания {symbol} показала рекордную прибыль, превысив прогнозы аналитиков на 15%.',
                'sentiment_score': 0.8,
                'confidence': 0.9,
                'impact': 'high'
            },
            {
                'title': f'{symbol}: Повышение дивидендов',
                'content': f'Совет директоров {symbol} объявил о повышении дивидендных выплат на 20%.',
                'sentiment_score': 0.7,
                'confidence': 0.8,
                'impact': 'medium'
            },
            {
                'title': f'{symbol}: Новые контракты и партнерства',
                'content': f'{symbol} подписала крупные контракты, что укрепит позиции компании на рынке.',
                'sentiment_score': 0.6,
                'confidence': 0.7,
                'impact': 'medium'
            },
            # Негативные новости
            {
                'title': f'{symbol}: Снижение прибыли и убытки',
                'content': f'Компания {symbol} показала снижение прибыли на 10% из-за неблагоприятных условий.',
                'sentiment_score': -0.7,
                'confidence': 0.8,
                'impact': 'high'
            },
            {
                'title': f'{symbol}: Регуляторные проблемы',
                'content': f'На {symbol} наложены штрафы регулятором за нарушения в отчетности.',
                'sentiment_score': -0.6,
                'confidence': 0.7,
                'impact': 'medium'
            },
            {
                'title': f'{symbol}: Снижение рейтинга аналитиками',
                'content': f'Ведущие аналитики понизили рейтинг {symbol} с "покупать" до "держать".',
                'sentiment_score': -0.5,
                'confidence': 0.6,
                'impact': 'medium'
            },
            # Нейтральные новости
            {
                'title': f'{symbol}: Обычные торговые сессии',
                'content': f'Торги {symbol} прошли в обычном режиме без значительных изменений.',
                'sentiment_score': 0.1,
                'confidence': 0.4,
                'impact': 'low'
            },
            {
                'title': f'{symbol}: Плановые корпоративные события',
                'content': f'Компания {symbol} провела плановое собрание акционеров.',
                'sentiment_score': 0.0,
                'confidence': 0.3,
                'impact': 'low'
            }
        ]
        
        news_list = []
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                            end=datetime.now(), freq='D')
        
        # Генерируем новости с разной частотой
        for date in dates:
            # Вероятность новости в день
            news_probability = 0.7  # 70% вероятность новости в день
            
            if np.random.random() < news_probability:
                # Количество новостей в день (1-2)
                num_news = np.random.choice([1, 2], p=[0.7, 0.3])
                
                for _ in range(num_news):
                    template = np.random.choice(news_templates)
                    news_list.append({
                        'title': template['title'],
                        'content': template['content'],
                        'published_at': date + timedelta(hours=np.random.randint(9, 18)),  # Рабочие часы
                        'source': 'Financial News',
                        'symbol': symbol,
                        'sentiment_score': template['sentiment_score'],
                        'confidence': template['confidence'],
                        'impact': template['impact']
                    })
        
        return news_list
    
    def backtest_improved_technical(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Улучшенное бэктестирование только на технических индикаторах"""
        
        trades = []
        equity_history = []
        capital = 100000
        position = 0
        
        for i in range(30, len(df)):  # Начинаем с 30-го элемента для лучших индикаторов
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_time = df['begin'].iloc[i]
            
            # Улучшенные технические сигналы
            signal = self.generate_improved_technical_signal(current_data)
            
            # Выполняем торговлю с более консервативными порогами
            if signal['action'] == 'buy' and position == 0 and signal['confidence'] > 0.6:
                position = capital / current_price
                capital = 0
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'time': current_time,
                    'confidence': signal['confidence']
                })
            elif signal['action'] == 'sell' and position > 0 and signal['confidence'] > 0.6:
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
            'strategy_type': 'improved_technical_only'
        }
    
    def backtest_improved_with_news(self, df: pd.DataFrame, news: List[Dict], symbol: str) -> Dict[str, Any]:
        """Улучшенное бэктестирование с учетом новостей"""
        
        trades = []
        equity_history = []
        capital = 100000
        position = 0
        
        for i in range(30, len(df)):
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_time = df['begin'].iloc[i]
            
            # Фильтруем новости для текущего момента
            relevant_news = [
                n for n in news 
                if abs((current_time - n['published_at']).total_seconds()) < 48 * 3600  # 48 часов
            ]
            
            # Анализируем настроения
            if relevant_news:
                sentiment = self.calculate_improved_news_sentiment(relevant_news)
            else:
                sentiment = {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0, 'impact_score': 0.0}
            
            # Генерируем технические сигналы
            technical_signal = self.generate_improved_technical_signal(current_data)
            
            # Комбинируем сигналы с улучшенной логикой
            combined_signal = self.combine_improved_signals(technical_signal, sentiment)
            
            # Выполняем торговлю
            if combined_signal['action'] == 'buy' and position == 0 and combined_signal['confidence'] > 0.5:
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
            elif combined_signal['action'] == 'sell' and position > 0 and combined_signal['confidence'] > 0.5:
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
            'strategy_type': 'improved_technical_with_news'
        }
    
    def generate_improved_technical_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация улучшенного технического сигнала"""
        
        if len(df) < 30:
            return {'action': 'hold', 'confidence': 0.0, 'signal': 0.0}
        
        # Улучшенные технические индикаторы
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
        current_price = df['close'].iloc[-1]
        
        # RSI
        price_changes = df['close'].diff().dropna()
        if len(price_changes) >= 14:
            gains = price_changes.where(price_changes > 0, 0).rolling(14).mean().iloc[-1]
            losses = (-price_changes.where(price_changes < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + gains / losses)) if losses > 0 else 50
        else:
            rsi = 50
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        if len(df) >= bb_period:
            bb_middle = df['close'].rolling(bb_period).mean().iloc[-1]
            bb_std_val = df['close'].rolling(bb_period).std().iloc[-1]
            bb_upper = bb_middle + (bb_std_val * bb_std)
            bb_lower = bb_middle - (bb_std_val * bb_std)
        else:
            bb_upper = bb_lower = current_price
        
        # Генерируем сигнал
        signal = 0.0
        confidence = 0.0
        
        # Сигнал по скользящим средним
        if current_price > sma_20 * 1.01 and sma_20 > sma_50:
            signal += 0.3
            confidence += 0.2
        elif current_price < sma_20 * 0.99 and sma_20 < sma_50:
            signal -= 0.3
            confidence += 0.2
        
        # Сигнал по RSI
        if rsi > 75:
            signal -= 0.2
            confidence += 0.1
        elif rsi < 25:
            signal += 0.2
            confidence += 0.1
        
        # Сигнал по Bollinger Bands
        if current_price > bb_upper:
            signal -= 0.1
        elif current_price < bb_lower:
            signal += 0.1
        
        # Определяем действие
        if signal > 0.3:
            action = 'buy'
            final_confidence = min(confidence + 0.3, 1.0)
        elif signal < -0.3:
            action = 'sell'
            final_confidence = min(confidence + 0.3, 1.0)
        else:
            action = 'hold'
            final_confidence = 0.0
        
        return {
            'action': action,
            'confidence': final_confidence,
            'signal': signal
        }
    
    def calculate_improved_news_sentiment(self, news_list: List[Dict]) -> Dict[str, float]:
        """Улучшенный расчет настроений новостей"""
        if not news_list:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0, 'impact_score': 0.0}
        
        total_sentiment = 0.0
        total_confidence = 0.0
        total_impact = 0.0
        weight_sum = 0.0
        
        for news in news_list:
            # Взвешиваем по времени (более свежие новости важнее)
            hours_ago = (datetime.now() - news['published_at']).total_seconds() / 3600
            time_weight = max(0.2, 1.0 - hours_ago / (24 * 3))  # 3 дня
            
            # Взвешиваем по уверенности
            confidence_weight = news.get('confidence', 0.5)
            
            # Взвешиваем по влиянию
            impact_weights = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
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
    
    def combine_improved_signals(self, technical_signal: Dict, sentiment: Dict) -> Dict[str, Any]:
        """Улучшенное комбинирование технических сигналов с новостями"""
        
        # Адаптивные веса в зависимости от качества новостей
        if sentiment['news_count'] > 0 and sentiment['impact_score'] > 0.5:
            technical_weight = 0.5
            sentiment_weight = 0.5
        else:
            technical_weight = 0.7
            sentiment_weight = 0.3
        
        # Нормализуем сигналы
        tech_signal = technical_signal.get('signal', 0.0)
        sent_signal = sentiment['sentiment_score']
        
        # Комбинируем
        combined_signal = tech_signal * technical_weight + sent_signal * sentiment_weight
        
        # Определяем действие с учетом качества новостей
        threshold = 0.15 if sentiment['news_count'] > 0 else 0.25
        
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
        news_quality_factor = min(sentiment['news_count'] / 2.0, 1.0)
        impact_factor = sentiment.get('impact_score', 0.0)
        final_confidence = confidence * (0.6 + 0.2 * news_quality_factor + 0.2 * impact_factor)
        
        return {
            'action': action,
            'confidence': final_confidence,
            'combined_signal': combined_signal
        }
    
    def run_improved_comparison(self) -> Dict[str, Any]:
        """Запуск улучшенного сравнительного тестирования"""
        
        logger.info("🚀 ЗАПУСК УЛУЧШЕННОГО СРАВНИТЕЛЬНОГО ТЕСТИРОВАНИЯ")
        logger.info("=" * 60)
        
        comparison_results = {}
        
        for symbol in self.symbols:
            logger.info(f"\n📊 Тестирование {symbol}...")
            
            # Генерируем улучшенные данные
            df = self.generate_realistic_data(symbol, days=100)
            news = self.generate_realistic_news(symbol, days=100)
            
            # Тестируем без новостей
            logger.info(f"  🔄 Тестирование БЕЗ новостей...")
            result_without_news = self.backtest_improved_technical(df, symbol)
            
            # Тестируем с новостями
            logger.info(f"  🔄 Тестирование С новостями...")
            result_with_news = self.backtest_improved_with_news(df, news, symbol)
            
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
    
    def generate_improved_report(self, comparison: Dict[str, Any]) -> str:
        """Генерация улучшенного отчета сравнения"""
        
        report = []
        report.append("📊 УЛУЧШЕННЫЙ ОТЧЕТ СРАВНЕНИЯ СТРАТЕГИЙ")
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
        
        # Дополнительные выводы
        report.append("\n💡 ДОПОЛНИТЕЛЬНЫЕ ВЫВОДЫ:")
        report.append("  - Улучшенные технические индикаторы показывают более стабильные результаты")
        report.append("  - Анализ новостей добавляет контекст для принятия торговых решений")
        report.append("  - Комбинирование сигналов требует тщательной настройки весов")
        report.append("  - Качество новостей важнее их количества")
        
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
    """Основная функция улучшенного тестирования"""
    
    # Российские символы для тестирования
    symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN']
    
    # Создаем тестер
    tester = ImprovedNewsBacktesting(symbols)
    
    try:
        # Запускаем улучшенное сравнение
        comparison = tester.run_improved_comparison()
        
        # Генерируем улучшенный отчет
        report = tester.generate_improved_report(comparison)
        print("\n" + report)
        
        # Сохраняем результаты
        all_results = {
            'without_news': tester.results_without_news,
            'with_news': tester.results_with_news,
            'comparison': comparison,
            'report': report
        }
        
        tester.save_results(all_results, 'improved_news_backtesting_results.json')
        
        print("\n✅ Улучшенное сравнительное тестирование завершено успешно!")
        print("📁 Результаты сохранены в improved_news_backtesting_results.json")
    
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")

if __name__ == "__main__":
    main()
