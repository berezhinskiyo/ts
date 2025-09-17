#!/usr/bin/env python3
"""
Интеграция анализа российских новостей с торговыми стратегиями
Замена NewsSentimentAnalyzer в advanced_tensortrade_robots.py на российскую версию
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Импортируем наш российский анализатор
from russian_news_analyzer import RussianNewsAnalyzer, RussianNewsItem

logger = logging.getLogger(__name__)

class RussianTradingStrategy:
    """Российская торговая стратегия с анализом новостей"""
    
    def __init__(self, symbols: List[str], config_file: str = "russian_news_config.json"):
        self.symbols = symbols
        self.news_analyzer = RussianNewsAnalyzer(config_file)
        
        # Параметры стратегии для российского рынка
        self.sentiment_weight = 0.4  # Новости важнее для российского рынка
        self.technical_weight = 0.3
        self.pattern_weight = 0.3
        
        # Пороги для торговых решений (адаптированы для российского рынка)
        self.buy_threshold = 0.15  # Более консервативные пороги
        self.sell_threshold = -0.15
        self.confidence_threshold = 0.35
        
        # Российские символы и их веса
        self.symbol_weights = {
            'SBER': 0.25,  # Сбербанк - самый ликвидный
            'GAZP': 0.20,  # Газпром
            'LKOH': 0.15,  # Лукойл
            'NVTK': 0.10,  # Новатэк
            'ROSN': 0.10,  # Роснефть
            'TATN': 0.08,  # Татнефть
            'MGNT': 0.07,  # Магнит
            'MTSS': 0.05   # МТС
        }
        
        logger.info(f"✅ Российская торговая стратегия инициализирована для {len(symbols)} символов")
    
    async def get_trading_signals(self, symbol: str, technical_signals: Dict, 
                                pattern_signals: Dict, days_back: int = 3) -> Dict[str, Any]:
        """Генерация торговых сигналов с учетом российских новостей"""
        
        try:
            # Получаем новости и анализируем настроения
            news = await self.news_analyzer.get_news_for_symbol(symbol, days_back)
            sentiment_analysis = self.news_analyzer.calculate_aggregate_sentiment(news)
            
            # Адаптируем веса для российского рынка
            symbol_weight = self.symbol_weights.get(symbol, 0.1)
            adjusted_sentiment_weight = self.sentiment_weight * symbol_weight * 2
            
            # Комбинируем сигналы
            combined_signal = self.combine_signals(
                sentiment_analysis,
                technical_signals,
                pattern_signals,
                adjusted_sentiment_weight
            )
            
            # Принимаем торговое решение
            trading_decision = self.make_trading_decision(combined_signal, sentiment_analysis, symbol)
            
            return {
                'symbol': symbol,
                'action': trading_decision['action'],
                'confidence': trading_decision['confidence'],
                'sentiment_score': sentiment_analysis['sentiment_score'],
                'sentiment_confidence': sentiment_analysis['confidence'],
                'news_count': sentiment_analysis['news_count'],
                'technical_signal': technical_signals.get('signal', 0.0),
                'pattern_signal': pattern_signals.get('signal', 0.0),
                'combined_signal': combined_signal,
                'reasoning': trading_decision['reasoning'],
                'recent_news': news[:3] if news else [],
                'market_impact': self.assess_market_impact(news)
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации сигналов для {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'hold',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def combine_signals(self, sentiment: Dict, technical: Dict, pattern: Dict, 
                       sentiment_weight: float = None) -> float:
        """Комбинирование различных типов сигналов с учетом российского рынка"""
        
        if sentiment_weight is None:
            sentiment_weight = self.sentiment_weight
        
        # Нормализуем сигналы к диапазону [-1, 1]
        sentiment_signal = np.clip(sentiment['sentiment_score'], -1, 1)
        technical_signal = np.clip(technical.get('signal', 0.0), -1, 1)
        pattern_signal = np.clip(pattern.get('signal', 0.0), -1, 1)
        
        # Взвешенная комбинация с учетом российского рынка
        combined = (
            sentiment_signal * sentiment_weight +
            technical_signal * self.technical_weight +
            pattern_signal * self.pattern_weight
        )
        
        return np.clip(combined, -1, 1)
    
    def make_trading_decision(self, combined_signal: float, sentiment: Dict, symbol: str) -> Dict[str, Any]:
        """Принятие торгового решения с учетом особенностей российского рынка"""
        
        # Адаптируем пороги для разных символов
        symbol_weight = self.symbol_weights.get(symbol, 0.1)
        buy_threshold = self.buy_threshold * (1 + symbol_weight)
        sell_threshold = self.sell_threshold * (1 + symbol_weight)
        
        # Базовое решение по сигналу
        if combined_signal > buy_threshold:
            base_action = 'buy'
            base_confidence = min(combined_signal, 1.0)
        elif combined_signal < sell_threshold:
            base_action = 'sell'
            base_confidence = min(abs(combined_signal), 1.0)
        else:
            base_action = 'hold'
            base_confidence = 0.0
        
        # Корректируем уверенность на основе качества новостей
        news_quality_factor = min(sentiment['news_count'] / 5.0, 1.0)  # Нормализуем к 5 новостям
        sentiment_confidence_factor = sentiment['confidence']
        
        # Дополнительный фактор для российского рынка
        russian_market_factor = 0.8 if sentiment['news_count'] > 0 else 0.5
        
        final_confidence = (base_confidence * news_quality_factor * 
                          sentiment_confidence_factor * russian_market_factor)
        
        # Генерируем обоснование решения
        reasoning = self.generate_russian_reasoning(combined_signal, sentiment, base_action, 
                                                  final_confidence, symbol)
        
        return {
            'action': base_action,
            'confidence': final_confidence,
            'reasoning': reasoning
        }
    
    def generate_russian_reasoning(self, signal: float, sentiment: Dict, action: str, 
                                 confidence: float, symbol: str) -> str:
        """Генерация обоснования торгового решения на русском языке"""
        
        reasoning_parts = []
        
        # Анализ сигнала
        if abs(signal) > 0.4:
            reasoning_parts.append(f"Сильный сигнал ({signal:.2f})")
        elif abs(signal) > 0.2:
            reasoning_parts.append(f"Умеренный сигнал ({signal:.2f})")
        else:
            reasoning_parts.append(f"Слабый сигнал ({signal:.2f})")
        
        # Анализ новостей
        if sentiment['news_count'] > 0:
            if sentiment['sentiment_score'] > 0.2:
                reasoning_parts.append(f"Позитивные новости ({sentiment['news_count']} статей)")
            elif sentiment['sentiment_score'] < -0.2:
                reasoning_parts.append(f"Негативные новости ({sentiment['news_count']} статей)")
            else:
                reasoning_parts.append(f"Нейтральные новости ({sentiment['news_count']} статей)")
        else:
            reasoning_parts.append("Нет свежих новостей")
        
        # Анализ уверенности
        if confidence > 0.7:
            reasoning_parts.append("Высокая уверенность")
        elif confidence > 0.4:
            reasoning_parts.append("Средняя уверенность")
        else:
            reasoning_parts.append("Низкая уверенность")
        
        # Специфика российского рынка
        if symbol in ['SBER', 'GAZP', 'LKOH']:
            reasoning_parts.append("Голубые фишки")
        elif symbol in ['NVTK', 'ROSN', 'TATN']:
            reasoning_parts.append("Нефтегазовый сектор")
        
        return f"{action.upper()}: " + ", ".join(reasoning_parts)
    
    def assess_market_impact(self, news: List[RussianNewsItem]) -> Dict[str, Any]:
        """Оценка влияния новостей на рынок"""
        
        if not news:
            return {
                'impact_level': 'low',
                'impact_score': 0.0,
                'key_themes': []
            }
        
        # Анализируем темы новостей
        themes = []
        impact_scores = []
        
        for news_item in news:
            # Простой анализ тем по ключевым словам
            text = (news_item.title + " " + news_item.content).lower()
            
            if any(word in text for word in ['санкции', 'санкция', 'ограничения']):
                themes.append('санкции')
                impact_scores.append(0.8)
            elif any(word in text for word in ['цб', 'центробанк', 'ставка', 'ключевая ставка']):
                themes.append('монетарная политика')
                impact_scores.append(0.7)
            elif any(word in text for word in ['нефть', 'газ', 'энергетика']):
                themes.append('энергетика')
                impact_scores.append(0.6)
            elif any(word in text for word in ['отчет', 'прибыль', 'выручка']):
                themes.append('корпоративные результаты')
                impact_scores.append(0.5)
            else:
                impact_scores.append(0.3)
        
        # Рассчитываем общий уровень влияния
        avg_impact = np.mean(impact_scores) if impact_scores else 0.0
        
        if avg_impact > 0.7:
            impact_level = 'high'
        elif avg_impact > 0.4:
            impact_level = 'medium'
        else:
            impact_level = 'low'
        
        return {
            'impact_level': impact_level,
            'impact_score': avg_impact,
            'key_themes': list(set(themes))
        }
    
    async def backtest_russian_strategy(self, historical_data: Dict[str, pd.DataFrame], 
                                      days_back: int = 7) -> Dict[str, Any]:
        """Бэктестирование российской стратегии с учетом новостей"""
        
        results = {}
        
        for symbol in self.symbols:
            if symbol not in historical_data:
                continue
            
            logger.info(f"🔄 Бэктестирование российской стратегии для {symbol}...")
            
            df = historical_data[symbol]
            trades = []
            equity_history = []
            capital = 100000  # Начальный капитал
            position = 0
            
            # Получаем новости для всего периода
            news = await self.news_analyzer.get_news_for_symbol(symbol, days_back)
            
            # Проходим по историческим данным
            for i in range(60, len(df)):  # Начинаем с 60-го элемента
                current_data = df.iloc[:i+1]
                current_price = df['close'].iloc[i]
                current_time = df['begin'].iloc[i]
                
                # Фильтруем новости для текущего момента
                relevant_news = [
                    n for n in news 
                    if abs((current_time - n.published_at).total_seconds()) < 24 * 3600
                ]
                
                if relevant_news:
                    sentiment = self.news_analyzer.calculate_aggregate_sentiment(relevant_news)
                else:
                    sentiment = {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0}
                
                # Генерируем технические сигналы
                technical_signals = self.generate_russian_technical_signals(current_data, symbol)
                pattern_signals = {'signal': 0.0}  # Заглушка для паттернов
                
                # Получаем торговый сигнал
                signal = await self.get_trading_signals(
                    symbol, technical_signals, pattern_signals, days_back=1
                )
                
                # Выполняем торговлю
                if signal['action'] == 'buy' and position == 0 and signal['confidence'] > self.confidence_threshold:
                    position = capital / current_price
                    capital = 0
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'time': current_time,
                        'confidence': signal['confidence'],
                        'sentiment': sentiment['sentiment_score'],
                        'news_count': sentiment['news_count'],
                        'market_impact': signal.get('market_impact', {})
                    })
                
                elif signal['action'] == 'sell' and position > 0 and signal['confidence'] > self.confidence_threshold:
                    capital = position * current_price
                    position = 0
                    trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'time': current_time,
                        'confidence': signal['confidence'],
                        'sentiment': sentiment['sentiment_score'],
                        'news_count': sentiment['news_count'],
                        'market_impact': signal.get('market_impact', {})
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
                'avg_sentiment': np.mean([t['sentiment'] for t in trades]) if trades else 0.0,
                'avg_news_count': np.mean([t['news_count'] for t in trades]) if trades else 0.0
            }
            
            logger.info(f"✅ {symbol}: Доходность={total_return:.2f}%, Просадка={max_drawdown:.2f}%, "
                       f"Сделок={len(trades)}, Средний сентимент={results[symbol]['avg_sentiment']:.3f}")
        
        return results
    
    def generate_russian_technical_signals(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Генерация технических сигналов для российского рынка"""
        
        if len(df) < 20:
            return {'signal': 0.0}
        
        # Простые технические индикаторы
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
        current_price = df['close'].iloc[-1]
        
        # RSI (упрощенный)
        price_changes = df['close'].diff().dropna()
        if len(price_changes) >= 14:
            gains = price_changes.where(price_changes > 0, 0).rolling(14).mean().iloc[-1]
            losses = (-price_changes.where(price_changes < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + gains / losses)) if losses > 0 else 50
        else:
            rsi = 50
        
        # Генерируем сигнал с учетом особенностей российского рынка
        signal = 0.0
        
        # Сигнал по скользящим средним
        if current_price > sma_20 * 1.01:  # Более консервативные пороги
            signal += 0.2
        elif current_price < sma_20 * 0.99:
            signal -= 0.2
        
        # Дополнительный сигнал по долгосрочной SMA
        if current_price > sma_50 * 1.02:
            signal += 0.1
        elif current_price < sma_50 * 0.98:
            signal -= 0.1
        
        # Сигнал по RSI (адаптированный для российского рынка)
        if rsi > 75:  # Более консервативные уровни
            signal -= 0.15
        elif rsi < 25:
            signal += 0.15
        
        # Дополнительный фактор для голубых фишек
        if symbol in ['SBER', 'GAZP', 'LKOH']:
            signal *= 1.1  # Увеличиваем уверенность для ликвидных акций
        
        return {'signal': np.clip(signal, -1, 1)}
    
    async def close(self):
        """Закрытие соединений"""
        await self.news_analyzer.close()

# Пример использования
async def main():
    """Пример использования российской торговой стратегии"""
    
    # Российские символы
    russian_symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN']
    
    # Создаем стратегию
    strategy = RussianTradingStrategy(russian_symbols)
    
    # Пример технических и паттерн сигналов
    technical_signals = {'signal': 0.2}
    pattern_signals = {'signal': 0.1}
    
    # Получаем торговые сигналы
    for symbol in russian_symbols:
        signal = await strategy.get_trading_signals(symbol, technical_signals, pattern_signals)
        
        print(f"\n📊 Торговый сигнал для {symbol}:")
        print(f"  Действие: {signal['action']}")
        print(f"  Уверенность: {signal['confidence']:.3f}")
        print(f"  Сентимент: {signal['sentiment_score']:.3f}")
        print(f"  Новостей: {signal['news_count']}")
        print(f"  Влияние на рынок: {signal.get('market_impact', {}).get('impact_level', 'unknown')}")
        print(f"  Обоснование: {signal['reasoning']}")
    
    # Закрываем соединения
    await strategy.close()

if __name__ == "__main__":
    asyncio.run(main())
