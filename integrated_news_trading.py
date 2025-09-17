#!/usr/bin/env python3
"""
Интеграция улучшенного анализатора новостей с торговыми стратегиями
Замена NewsSentimentAnalyzer в advanced_tensortrade_robots.py
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Импортируем наш улучшенный анализатор
from enhanced_news_analyzer import EnhancedNewsSentimentAnalyzer, NewsItem

logger = logging.getLogger(__name__)

class IntegratedNewsTradingStrategy:
    """Интегрированная торговая стратегия с анализом новостей"""
    
    def __init__(self, symbols: List[str], config_file: str = "news_config.json"):
        self.symbols = symbols
        self.news_analyzer = EnhancedNewsSentimentAnalyzer(config_file)
        
        # Параметры стратегии
        self.sentiment_weight = 0.3
        self.technical_weight = 0.4
        self.pattern_weight = 0.3
        
        # Пороги для торговых решений
        self.buy_threshold = 0.2
        self.sell_threshold = -0.2
        self.confidence_threshold = 0.4
        
        logger.info(f"✅ Интегрированная торговая стратегия инициализирована для {len(symbols)} символов")
    
    async def get_trading_signals(self, symbol: str, technical_signals: Dict, 
                                pattern_signals: Dict, days_back: int = 3) -> Dict[str, Any]:
        """Генерация торговых сигналов с учетом новостей"""
        
        try:
            # Получаем новости и анализируем настроения
            news = await self.news_analyzer.get_news_for_symbol(symbol, days_back)
            sentiment_analysis = self.news_analyzer.calculate_aggregate_sentiment(news)
            
            # Комбинируем сигналы
            combined_signal = self.combine_signals(
                sentiment_analysis,
                technical_signals,
                pattern_signals
            )
            
            # Принимаем торговое решение
            trading_decision = self.make_trading_decision(combined_signal, sentiment_analysis)
            
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
                'recent_news': news[:3] if news else []  # Последние 3 новости
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации сигналов для {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'hold',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def combine_signals(self, sentiment: Dict, technical: Dict, pattern: Dict) -> float:
        """Комбинирование различных типов сигналов"""
        
        # Нормализуем сигналы к диапазону [-1, 1]
        sentiment_signal = np.clip(sentiment['sentiment_score'], -1, 1)
        technical_signal = np.clip(technical.get('signal', 0.0), -1, 1)
        pattern_signal = np.clip(pattern.get('signal', 0.0), -1, 1)
        
        # Взвешенная комбинация
        combined = (
            sentiment_signal * self.sentiment_weight +
            technical_signal * self.technical_weight +
            pattern_signal * self.pattern_weight
        )
        
        return np.clip(combined, -1, 1)
    
    def make_trading_decision(self, combined_signal: float, sentiment: Dict) -> Dict[str, Any]:
        """Принятие торгового решения на основе комбинированного сигнала"""
        
        # Базовое решение по сигналу
        if combined_signal > self.buy_threshold:
            base_action = 'buy'
            base_confidence = min(combined_signal, 1.0)
        elif combined_signal < self.sell_threshold:
            base_action = 'sell'
            base_confidence = min(abs(combined_signal), 1.0)
        else:
            base_action = 'hold'
            base_confidence = 0.0
        
        # Корректируем уверенность на основе качества новостей
        news_quality_factor = min(sentiment['news_count'] / 10.0, 1.0)  # Нормализуем к 10 новостям
        sentiment_confidence_factor = sentiment['confidence']
        
        final_confidence = base_confidence * news_quality_factor * sentiment_confidence_factor
        
        # Генерируем обоснование решения
        reasoning = self.generate_reasoning(combined_signal, sentiment, base_action, final_confidence)
        
        return {
            'action': base_action,
            'confidence': final_confidence,
            'reasoning': reasoning
        }
    
    def generate_reasoning(self, signal: float, sentiment: Dict, action: str, confidence: float) -> str:
        """Генерация обоснования торгового решения"""
        
        reasoning_parts = []
        
        # Анализ сигнала
        if abs(signal) > 0.5:
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
        
        return f"{action.upper()}: " + ", ".join(reasoning_parts)
    
    async def backtest_with_news(self, historical_data: Dict[str, pd.DataFrame], 
                               days_back: int = 7) -> Dict[str, Any]:
        """Бэктестирование стратегии с учетом новостей"""
        
        results = {}
        
        for symbol in self.symbols:
            if symbol not in historical_data:
                continue
            
            logger.info(f"🔄 Бэктестирование {symbol} с анализом новостей...")
            
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
                    if abs((current_time - n.published_at).total_seconds()) < 24 * 3600  # За последние 24 часа
                ]
                
                if relevant_news:
                    sentiment = self.news_analyzer.calculate_aggregate_sentiment(relevant_news)
                else:
                    sentiment = {'sentiment_score': 0.0, 'confidence': 0.0, 'news_count': 0}
                
                # Генерируем технические сигналы (упрощенно)
                technical_signals = self.generate_technical_signals(current_data)
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
                        'news_count': sentiment['news_count']
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
            
            results[symbol] = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'trades': trades,
                'equity_history': equity_history,
                'final_equity': final_equity
            }
            
            logger.info(f"✅ {symbol}: Доходность={total_return:.2f}%, Просадка={max_drawdown:.2f}%, Сделок={len(trades)}")
        
        return results
    
    def generate_technical_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Генерация технических сигналов (упрощенная версия)"""
        
        if len(df) < 20:
            return {'signal': 0.0}
        
        # Простые технические индикаторы
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # RSI (упрощенный)
        price_changes = df['close'].diff().dropna()
        gains = price_changes.where(price_changes > 0, 0).rolling(14).mean().iloc[-1]
        losses = (-price_changes.where(price_changes < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gains / losses)) if losses > 0 else 50
        
        # Генерируем сигнал
        signal = 0.0
        
        # Сигнал по скользящей средней
        if current_price > sma_20 * 1.02:  # Цена на 2% выше SMA
            signal += 0.3
        elif current_price < sma_20 * 0.98:  # Цена на 2% ниже SMA
            signal -= 0.3
        
        # Сигнал по RSI
        if rsi > 70:  # Перекупленность
            signal -= 0.2
        elif rsi < 30:  # Перепроданность
            signal += 0.2
        
        return {'signal': np.clip(signal, -1, 1)}
    
    async def close(self):
        """Закрытие соединений"""
        await self.news_analyzer.close()

# Пример использования
async def main():
    """Пример использования интегрированной стратегии"""
    
    # Создаем стратегию
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    strategy = IntegratedNewsTradingStrategy(symbols)
    
    # Пример технических и паттерн сигналов
    technical_signals = {'signal': 0.3}
    pattern_signals = {'signal': 0.1}
    
    # Получаем торговые сигналы
    for symbol in symbols:
        signal = await strategy.get_trading_signals(symbol, technical_signals, pattern_signals)
        
        print(f"\n📊 Торговый сигнал для {symbol}:")
        print(f"  Действие: {signal['action']}")
        print(f"  Уверенность: {signal['confidence']:.3f}")
        print(f"  Сентимент: {signal['sentiment_score']:.3f}")
        print(f"  Новостей: {signal['news_count']}")
        print(f"  Обоснование: {signal['reasoning']}")
    
    # Закрываем соединения
    await strategy.close()

if __name__ == "__main__":
    asyncio.run(main())
