#!/usr/bin/env python3
"""
Демонстрационный скрипт для анализа российских новостей
Показывает возможности системы без необходимости настройки API
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemoRussianNewsAnalyzer:
    """Демонстрационный анализатор российских новостей"""
    
    def __init__(self):
        self.sample_news = self.generate_sample_news()
        self.sentiment_keywords = {
            'positive': [
                'рост', 'повышение', 'увеличение', 'прибыль', 'успех', 'победа',
                'позитив', 'хорошо', 'отлично', 'прекрасно', 'замечательно',
                'стабильность', 'устойчивость', 'развитие', 'прогресс',
                'рекорд', 'достижение', 'улучшение', 'подъем'
            ],
            'negative': [
                'падение', 'снижение', 'уменьшение', 'убыток', 'провал', 'поражение',
                'негатив', 'плохо', 'ужасно', 'катастрофа', 'кризис',
                'нестабильность', 'риск', 'опасность', 'проблема',
                'снижение', 'ухудшение', 'спад', 'деградация'
            ]
        }
    
    def generate_sample_news(self) -> Dict[str, List[Dict]]:
        """Генерация примеров российских новостей"""
        
        symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN']
        news_data = {}
        
        for symbol in symbols:
            news_data[symbol] = [
                {
                    'title': f'{symbol}: Аналитики повышают прогнозы на фоне роста прибыли',
                    'content': f'Ведущие аналитики пересматривают свои прогнозы по акциям {symbol} в сторону повышения. Компания показала рекордную прибыль в последнем квартале, что говорит о стабильном развитии бизнеса.',
                    'published_at': datetime.now() - timedelta(hours=2),
                    'source': 'РБК',
                    'url': f'https://www.rbc.ru/finances/{symbol}',
                    'sentiment_score': 0.7,
                    'confidence': 0.8
                },
                {
                    'title': f'Торги {symbol} на Мосбирже показывают положительную динамику',
                    'content': f'Акции {symbol} демонстрируют устойчивый рост на фоне позитивных макроэкономических показателей. Инвесторы проявляют повышенный интерес к бумагам компании.',
                    'published_at': datetime.now() - timedelta(hours=5),
                    'source': 'Ведомости',
                    'url': f'https://www.vedomости.ru/finance/{symbol}',
                    'sentiment_score': 0.6,
                    'confidence': 0.7
                },
                {
                    'title': f'{symbol}: Компания объявляет о новых инвестиционных планах',
                    'content': f'Руководство {symbol} представило амбициозные планы по развитию бизнеса. Компания планирует увеличить инвестиции в цифровые технологии и расширить присутствие на международных рынках.',
                    'published_at': datetime.now() - timedelta(hours=8),
                    'source': 'Коммерсантъ',
                    'url': f'https://www.kommersant.ru/finance/{symbol}',
                    'sentiment_score': 0.5,
                    'confidence': 0.6
                },
                {
                    'title': f'Эксперты отмечают высокую ликвидность акций {symbol}',
                    'content': f'Финансовые эксперты подчеркивают высокую ликвидность и стабильность акций {symbol}. Бумаги компании пользуются популярностью как у частных, так и у институциональных инвесторов.',
                    'published_at': datetime.now() - timedelta(hours=12),
                    'source': 'ТАСС',
                    'url': f'https://tass.ru/ekonomika/{symbol}',
                    'sentiment_score': 0.4,
                    'confidence': 0.5
                }
            ]
        
        return news_data
    
    def analyze_sentiment_keywords(self, text: str) -> Dict[str, float]:
        """Анализ настроения по ключевым словам"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return {'score': 0.0, 'confidence': 0.0}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        sentiment_score = positive_ratio - negative_ratio
        confidence = min(positive_ratio + negative_ratio, 1.0)
        
        return {
            'score': sentiment_score,
            'confidence': confidence
        }
    
    def calculate_aggregate_sentiment(self, news_list: List[Dict]) -> Dict[str, float]:
        """Расчет агрегированного индекса настроений"""
        if not news_list:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'news_count': 0,
                'positive_ratio': 0.5
            }
        
        total_sentiment = 0.0
        total_confidence = 0.0
        weight_sum = 0.0
        positive_count = 0
        
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
            
            if news['sentiment_score'] > 0:
                positive_count += 1
        
        avg_sentiment = total_sentiment / weight_sum if weight_sum > 0 else 0.0
        avg_confidence = total_confidence / weight_sum if weight_sum > 0 else 0.0
        positive_ratio = positive_count / len(news_list)
        
        return {
            'sentiment_score': avg_sentiment,
            'confidence': avg_confidence,
            'news_count': len(news_list),
            'positive_ratio': positive_ratio
        }
    
    def generate_trading_signal(self, sentiment: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """Генерация торгового сигнала на основе настроений"""
        
        sentiment_score = sentiment['sentiment_score']
        confidence = sentiment['confidence']
        news_count = sentiment['news_count']
        
        # Определяем действие
        if sentiment_score > 0.2 and confidence > 0.4:
            action = 'buy'
            signal_strength = min(sentiment_score, 1.0)
        elif sentiment_score < -0.2 and confidence > 0.4:
            action = 'sell'
            signal_strength = min(abs(sentiment_score), 1.0)
        else:
            action = 'hold'
            signal_strength = 0.0
        
        # Корректируем уверенность на основе количества новостей
        news_quality_factor = min(news_count / 5.0, 1.0)
        final_confidence = signal_strength * confidence * news_quality_factor
        
        # Генерируем обоснование
        reasoning_parts = []
        
        if sentiment_score > 0.2:
            reasoning_parts.append("Позитивные новости")
        elif sentiment_score < -0.2:
            reasoning_parts.append("Негативные новости")
        else:
            reasoning_parts.append("Нейтральные новости")
        
        reasoning_parts.append(f"{news_count} статей")
        
        if confidence > 0.7:
            reasoning_parts.append("Высокая уверенность")
        elif confidence > 0.4:
            reasoning_parts.append("Средняя уверенность")
        else:
            reasoning_parts.append("Низкая уверенность")
        
        if symbol in ['SBER', 'GAZP', 'LKOH']:
            reasoning_parts.append("Голубые фишки")
        
        reasoning = f"{action.upper()}: " + ", ".join(reasoning_parts)
        
        return {
            'action': action,
            'confidence': final_confidence,
            'reasoning': reasoning,
            'sentiment_score': sentiment_score,
            'news_count': news_count
        }
    
    async def demo_analysis(self):
        """Демонстрация анализа российских новостей"""
        
        print("🚀 ДЕМОНСТРАЦИЯ АНАЛИЗА РОССИЙСКИХ НОВОСТЕЙ")
        print("=" * 60)
        
        symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN']
        
        for symbol in symbols:
            print(f"\n📊 Анализ для {symbol}")
            print("-" * 40)
            
            # Получаем новости
            news = self.sample_news[symbol]
            
            # Анализируем настроения
            sentiment = self.calculate_aggregate_sentiment(news)
            
            # Генерируем торговый сигнал
            signal = self.generate_trading_signal(sentiment, symbol)
            
            # Выводим результаты
            print(f"📰 Новостей: {sentiment['news_count']}")
            print(f"📈 Sentiment Score: {sentiment['sentiment_score']:.3f}")
            print(f"🎯 Confidence: {sentiment['confidence']:.3f}")
            print(f"📊 Positive Ratio: {sentiment['positive_ratio']:.3f}")
            print(f"💡 Trading Signal: {signal['action']}")
            print(f"🔍 Signal Confidence: {signal['confidence']:.3f}")
            print(f"📝 Reasoning: {signal['reasoning']}")
            
            # Показываем примеры новостей
            print(f"\n📰 Примеры новостей:")
            for i, news_item in enumerate(news[:2], 1):
                print(f"  {i}. {news_item['title']}")
                print(f"     Источник: {news_item['source']}")
                print(f"     Sentiment: {news_item['sentiment_score']:.2f}")
                print(f"     Время: {news_item['published_at'].strftime('%H:%M')}")
        
        print(f"\n✅ Демонстрация завершена!")
        print(f"💡 Для работы с реальными данными настройте API ключи в конфигурации")

async def main():
    """Основная функция демонстрации"""
    
    demo = DemoRussianNewsAnalyzer()
    await demo.demo_analysis()

if __name__ == "__main__":
    asyncio.run(main())
