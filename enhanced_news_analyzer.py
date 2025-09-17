#!/usr/bin/env python3
"""
Улучшенный анализатор новостей с интеграцией реальных API
Поддерживает множественные источники новостей для тестирования и продуктивной работы
"""

import os
import sys
import json
import time
import logging
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import sqlite3
from pathlib import Path

# NLP импорты
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers не установлен. Установите: pip install transformers torch")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """Структура новости"""
    title: str
    content: str
    published_at: datetime
    source: str
    url: str = ""
    symbol: str = ""
    sentiment_score: float = 0.0
    confidence: float = 0.0

class NewsCache:
    """Кэш для новостей с SQLite"""
    
    def __init__(self, cache_file: str = "news_cache.db"):
        self.cache_file = cache_file
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных кэша"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_cache (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                title TEXT,
                content TEXT,
                published_at TEXT,
                source TEXT,
                url TEXT,
                sentiment_score REAL,
                confidence REAL,
                cached_at TEXT
            )
        ''')
        
        # Создаем индекс для быстрого поиска
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_time 
            ON news_cache(symbol, published_at)
        ''')
        
        conn.commit()
        conn.close()
    
    def get_cache_key(self, symbol: str, days_back: int) -> str:
        """Генерация ключа кэша"""
        return hashlib.md5(f"{symbol}_{days_back}_{datetime.now().strftime('%Y-%m-%d')}".encode()).hexdigest()
    
    def get_cached_news(self, symbol: str, days_back: int) -> List[NewsItem]:
        """Получение кэшированных новостей"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        cursor.execute('''
            SELECT title, content, published_at, source, url, sentiment_score, confidence
            FROM news_cache 
            WHERE symbol = ? AND published_at >= ?
            ORDER BY published_at DESC
        ''', (symbol, cutoff_time.isoformat()))
        
        news_items = []
        for row in cursor.fetchall():
            news_items.append(NewsItem(
                title=row[0],
                content=row[1],
                published_at=datetime.fromisoformat(row[2]),
                source=row[3],
                url=row[4],
                sentiment_score=row[5],
                confidence=row[6]
            ))
        
        conn.close()
        return news_items
    
    def cache_news(self, symbol: str, news_items: List[NewsItem]):
        """Кэширование новостей"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        for news in news_items:
            news_id = hashlib.md5(f"{news.title}_{news.published_at}".encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO news_cache 
                (id, symbol, title, content, published_at, source, url, sentiment_score, confidence, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                news_id, symbol, news.title, news.content, 
                news.published_at.isoformat(), news.source, news.url,
                news.sentiment_score, news.confidence, datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()

class NewsProvider(ABC):
    """Абстрактный класс для провайдеров новостей"""
    
    @abstractmethod
    async def get_news(self, symbol: str, days_back: int = 7) -> List[NewsItem]:
        """Получение новостей для символа"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Название провайдера"""
        pass

class NewsAPIProvider(NewsProvider):
    """Провайдер NewsAPI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.session = None
    
    async def get_news(self, symbol: str, days_back: int = 7) -> List[NewsItem]:
        """Получение новостей через NewsAPI"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Формируем запрос
        params = {
            'q': f'{symbol} OR "{symbol}" stock OR "{symbol}" shares',
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
            'apiKey': self.api_key,
            'pageSize': 50
        }
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    news_items = []
                    
                    for article in data.get('articles', []):
                        if article.get('title') and article.get('content'):
                            news_items.append(NewsItem(
                                title=article['title'],
                                content=article['content'] or article['description'] or '',
                                published_at=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                                source=article['source']['name'],
                                url=article['url'],
                                symbol=symbol
                            ))
                    
                    return news_items
                else:
                    logger.error(f"NewsAPI error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Error fetching news from NewsAPI: {e}")
            return []
    
    def get_name(self) -> str:
        return "NewsAPI"

class AlphaVantageProvider(NewsProvider):
    """Провайдер Alpha Vantage News & Sentiment"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
    
    async def get_news(self, symbol: str, days_back: int = 7) -> List[NewsItem]:
        """Получение новостей через Alpha Vantage"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': self.api_key,
            'limit': 50
        }
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    news_items = []
                    
                    for article in data.get('feed', []):
                        if article.get('title') and article.get('summary'):
                            # Получаем сентимент для данного тикера
                            sentiment_score = 0.0
                            confidence = 0.0
                            
                            for ticker_sentiment in article.get('ticker_sentiment', []):
                                if ticker_sentiment['ticker'] == symbol:
                                    sentiment_score = float(ticker_sentiment['relevance_score'])
                                    confidence = float(ticker_sentiment['ticker_sentiment_score'])
                                    break
                            
                            news_items.append(NewsItem(
                                title=article['title'],
                                content=article['summary'],
                                published_at=datetime.fromisoformat(article['time_published'].replace('Z', '+00:00')),
                                source=article['source'],
                                url=article['url'],
                                symbol=symbol,
                                sentiment_score=sentiment_score,
                                confidence=confidence
                            ))
                    
                    return news_items
                else:
                    logger.error(f"Alpha Vantage error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Error fetching news from Alpha Vantage: {e}")
            return []
    
    def get_name(self) -> str:
        return "Alpha Vantage"

class YahooFinanceProvider(NewsProvider):
    """Провайдер Yahoo Finance (неофициальный)"""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v1/finance/search"
        self.session = None
    
    async def get_news(self, symbol: str, days_back: int = 7) -> List[NewsItem]:
        """Получение новостей через Yahoo Finance"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        params = {
            'q': symbol,
            'quotesCount': 1,
            'newsCount': 50
        }
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    news_items = []
                    
                    for article in data.get('news', []):
                        if article.get('title') and article.get('summary'):
                            news_items.append(NewsItem(
                                title=article['title'],
                                content=article['summary'],
                                published_at=datetime.fromtimestamp(article['providerPublishTime']),
                                source=article['publisher'],
                                url=article['link'],
                                symbol=symbol
                            ))
                    
                    return news_items
                else:
                    logger.error(f"Yahoo Finance error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Error fetching news from Yahoo Finance: {e}")
            return []
    
    def get_name(self) -> str:
        return "Yahoo Finance"

class EnhancedNewsSentimentAnalyzer:
    """Улучшенный анализатор новостей с множественными источниками"""
    
    def __init__(self, config_file: str = "news_config.json"):
        self.config = self.load_config(config_file)
        self.cache = NewsCache()
        self.providers = self.init_providers()
        self.sentiment_pipeline = self.init_sentiment_models()
        
        logger.info(f"✅ Enhanced News Analyzer инициализирован с {len(self.providers)} провайдерами")
    
    def load_config(self, config_file: str) -> Dict:
        """Загрузка конфигурации"""
        default_config = {
            "providers": {
                "newsapi": {"enabled": False, "api_key": ""},
                "alphavantage": {"enabled": False, "api_key": ""},
                "yahoofinance": {"enabled": True, "api_key": ""}
            },
            "cache": {
                "enabled": True,
                "ttl_hours": 1
            },
            "sentiment": {
                "use_financial_model": True,
                "confidence_threshold": 0.3
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации: {e}")
        
        return default_config
    
    def init_providers(self) -> List[NewsProvider]:
        """Инициализация провайдеров новостей"""
        providers = []
        
        if self.config["providers"]["newsapi"]["enabled"] and self.config["providers"]["newsapi"]["api_key"]:
            providers.append(NewsAPIProvider(self.config["providers"]["newsapi"]["api_key"]))
        
        if self.config["providers"]["alphavantage"]["enabled"] and self.config["providers"]["alphavantage"]["api_key"]:
            providers.append(AlphaVantageProvider(self.config["providers"]["alphavantage"]["api_key"]))
        
        if self.config["providers"]["yahoofinance"]["enabled"]:
            providers.append(YahooFinanceProvider())
        
        return providers
    
    def init_sentiment_models(self):
        """Инициализация моделей анализа настроений"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers не доступен, анализ настроений отключен")
            return None
        
        try:
            # Используем FinBERT для финансовых новостей
            if self.config["sentiment"]["use_financial_model"]:
                pipeline = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    return_all_scores=True
                )
            else:
                pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
            
            logger.info("✅ Модель анализа настроений загружена")
            return pipeline
        
        except Exception as e:
            logger.error(f"Ошибка загрузки модели анализа настроений: {e}")
            return None
    
    async def get_news_for_symbol(self, symbol: str, days_back: int = 7) -> List[NewsItem]:
        """Получение новостей для символа из всех доступных источников"""
        
        # Проверяем кэш
        if self.config["cache"]["enabled"]:
            cached_news = self.cache.get_cached_news(symbol, days_back)
            if cached_news:
                logger.info(f"📰 Найдено {len(cached_news)} кэшированных новостей для {symbol}")
                return cached_news
        
        # Получаем новости из всех провайдеров
        all_news = []
        
        for provider in self.providers:
            try:
                logger.info(f"🔍 Получение новостей для {symbol} из {provider.get_name()}")
                news = await provider.get_news(symbol, days_back)
                all_news.extend(news)
                logger.info(f"📰 Получено {len(news)} новостей из {provider.get_name()}")
                
                # Небольшая задержка между запросами
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Ошибка получения новостей из {provider.get_name()}: {e}")
        
        # Удаляем дубликаты по заголовку
        unique_news = []
        seen_titles = set()
        
        for news in all_news:
            title_hash = hashlib.md5(news.title.lower().encode()).hexdigest()
            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                unique_news.append(news)
        
        # Анализируем настроения для новостей без предварительного анализа
        for news in unique_news:
            if news.sentiment_score == 0.0 and self.sentiment_pipeline:
                sentiment = self.analyze_sentiment(news.title + " " + news.content)
                news.sentiment_score = sentiment['score']
                news.confidence = sentiment['confidence']
        
        # Кэшируем результаты
        if self.config["cache"]["enabled"] and unique_news:
            self.cache.cache_news(symbol, unique_news)
        
        logger.info(f"📊 Итого получено {len(unique_news)} уникальных новостей для {symbol}")
        return unique_news
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Анализ настроения текста"""
        if not text or not self.sentiment_pipeline:
            return {'score': 0.0, 'confidence': 0.0}
        
        try:
            results = self.sentiment_pipeline(text[:512])  # Ограничиваем длину
            
            # Преобразуем результаты в числовой формат
            positive_score = 0.0
            negative_score = 0.0
            
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label or 'bullish' in label:
                    positive_score = score
                elif 'negative' in label or 'bearish' in label:
                    negative_score = score
            
            # Рассчитываем общий сентимент (-1 до 1)
            sentiment_score = positive_score - negative_score
            confidence = max(positive_score, negative_score)
            
            return {
                'score': sentiment_score,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа настроения: {e}")
            return {'score': 0.0, 'confidence': 0.0}
    
    def calculate_aggregate_sentiment(self, news_list: List[NewsItem]) -> Dict[str, float]:
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
            # Взвешиваем по времени (более свежие новости важнее)
            hours_ago = (datetime.now() - news.published_at).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - hours_ago / (24 * 7))  # За неделю вес падает до 0.1
            
            # Взвешиваем по уверенности
            confidence_weight = news.confidence if news.confidence > 0 else 0.5
            
            # Общий вес
            weight = time_weight * confidence_weight
            
            total_sentiment += news.sentiment_score * weight
            total_confidence += news.confidence * weight
            weight_sum += weight
            
            if news.sentiment_score > 0:
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
    
    async def get_market_sentiment(self, symbols: List[str], days_back: int = 7) -> Dict[str, Dict[str, float]]:
        """Получение настроений рынка для списка символов"""
        results = {}
        
        for symbol in symbols:
            try:
                news = await self.get_news_for_symbol(symbol, days_back)
                sentiment = self.calculate_aggregate_sentiment(news)
                results[symbol] = sentiment
                
                logger.info(f"📈 {symbol}: Sentiment={sentiment['sentiment_score']:.3f}, "
                          f"Confidence={sentiment['confidence']:.3f}, "
                          f"News={sentiment['news_count']}")
                
            except Exception as e:
                logger.error(f"Ошибка получения настроений для {symbol}: {e}")
                results[symbol] = {
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'news_count': 0,
                    'positive_ratio': 0.5
                }
        
        return results
    
    async def close(self):
        """Закрытие соединений"""
        for provider in self.providers:
            if hasattr(provider, 'session') and provider.session:
                await provider.session.close()

# Пример использования
async def main():
    """Пример использования улучшенного анализатора новостей"""
    
    # Создаем анализатор
    analyzer = EnhancedNewsSentimentAnalyzer()
    
    # Получаем настроения для списка символов
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    sentiments = await analyzer.get_market_sentiment(symbols, days_back=3)
    
    # Выводим результаты
    print("\n📊 АНАЛИЗ НАСТРОЕНИЙ РЫНКА")
    print("=" * 50)
    
    for symbol, sentiment in sentiments.items():
        print(f"{symbol}:")
        print(f"  Sentiment Score: {sentiment['sentiment_score']:.3f}")
        print(f"  Confidence: {sentiment['confidence']:.3f}")
        print(f"  News Count: {sentiment['news_count']}")
        print(f"  Positive Ratio: {sentiment['positive_ratio']:.3f}")
        print()
    
    # Закрываем соединения
    await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())
