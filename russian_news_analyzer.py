#!/usr/bin/env python3
"""
Анализатор российских финансовых новостей с поддержкой Telegram каналов
Интеграция с MOEX, российскими СМИ и Telegram каналами
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
import re

# Telegram импорты
try:
    from telethon import TelegramClient, events
    from telethon.tl.types import Channel, Chat
    TELETHON_AVAILABLE = True
except ImportError:
    TELETHON_AVAILABLE = False
    print("⚠️ Telethon не установлен. Установите: pip install telethon")

# NLP импорты для русского языка
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
class RussianNewsItem:
    """Структура российской новости"""
    title: str
    content: str
    published_at: datetime
    source: str
    url: str = ""
    symbol: str = ""
    sentiment_score: float = 0.0
    confidence: float = 0.0
    language: str = "ru"

class MOEXNewsProvider:
    """Провайдер новостей MOEX (Московская биржа)"""
    
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss"
        self.session = None
    
    async def get_news(self, symbol: str = None, days_back: int = 7) -> List[RussianNewsItem]:
        """Получение новостей с MOEX"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # MOEX не предоставляет прямого API для новостей, но можно парсить их сайт
        # Здесь используем заглушку, в реальности нужно парсить moex.com
        news_items = []
        
        try:
            # Примеры новостей MOEX (в реальности нужно парсить сайт)
            sample_news = [
                {
                    'title': f'Торги {symbol} на Московской бирже показывают стабильность',
                    'content': f'Акции {symbol} демонстрируют устойчивые результаты на фоне позитивных макроэкономических показателей...',
                    'published_at': datetime.now() - timedelta(hours=2),
                    'source': 'MOEX',
                    'url': f'https://www.moex.com/news/{symbol}'
                }
            ]
            
            for news in sample_news:
                news_items.append(RussianNewsItem(
                    title=news['title'],
                    content=news['content'],
                    published_at=news['published_at'],
                    source=news['source'],
                    url=news['url'],
                    symbol=symbol or 'MOEX'
                ))
            
        except Exception as e:
            logger.error(f"Ошибка получения новостей MOEX: {e}")
        
        return news_items

class RussianMediaProvider:
    """Провайдер российских финансовых СМИ"""
    
    def __init__(self):
        self.sources = {
            'rbc': 'https://www.rbc.ru/finances/',
            'vedomosti': 'https://www.vedomosti.ru/finance',
            'kommersant': 'https://www.kommersant.ru/finance',
            'tass': 'https://tass.ru/ekonomika'
        }
        self.session = None
    
    async def get_news(self, symbol: str, days_back: int = 7) -> List[RussianNewsItem]:
        """Получение новостей из российских СМИ"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        news_items = []
        
        # В реальности здесь был бы парсинг сайтов
        # Для демонстрации используем примеры
        sample_news = [
            {
                'title': f'{symbol}: Аналитики повышают прогнозы',
                'content': f'Ведущие аналитики пересматривают свои прогнозы по акциям {symbol} в сторону повышения...',
                'published_at': datetime.now() - timedelta(hours=1),
                'source': 'РБК',
                'url': f'https://www.rbc.ru/finances/{symbol}'
            },
            {
                'title': f'Рынок акций: {symbol} показывает рост',
                'content': f'Акции {symbol} демонстрируют положительную динамику на фоне улучшения финансовых показателей...',
                'published_at': datetime.now() - timedelta(hours=3),
                'source': 'Ведомости',
                'url': f'https://www.vedomosti.ru/finance/{symbol}'
            }
        ]
        
        for news in sample_news:
            news_items.append(RussianNewsItem(
                title=news['title'],
                content=news['content'],
                published_at=news['published_at'],
                source=news['source'],
                url=news['url'],
                symbol=symbol
            ))
        
        return news_items

class TelegramChannelProvider:
    """Провайдер новостей из Telegram каналов"""
    
    def __init__(self, api_id: str, api_hash: str, phone: str = None):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.client = None
        self.channels = {
            'rbc_finance': '@rbc_finance',
            'vedomosti': '@vedomosti',
            'kommersant': '@kommersant',
            'tass_economics': '@tass_economics',
            'moex_official': '@moex_official',
            'investing_ru': '@investing_ru'
        }
    
    async def initialize(self):
        """Инициализация Telegram клиента"""
        if not TELETHON_AVAILABLE:
            logger.error("Telethon не установлен")
            return False
        
        try:
            self.client = TelegramClient('russian_news_bot', self.api_id, self.api_hash)
            await self.client.start(phone=self.phone)
            logger.info("✅ Telegram клиент инициализирован")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации Telegram: {e}")
            return False
    
    async def get_channel_history(self, channel_username: str, limit: int = 100) -> List[RussianNewsItem]:
        """Получение истории сообщений канала"""
        if not self.client:
            return []
        
        news_items = []
        
        try:
            # Получаем канал
            channel = await self.client.get_entity(channel_username)
            
            # Получаем сообщения
            messages = await self.client.get_messages(channel, limit=limit)
            
            for message in messages:
                if message.text and len(message.text) > 20:  # Фильтруем короткие сообщения
                    # Определяем, является ли сообщение финансовой новостью
                    if self.is_financial_news(message.text):
                        news_items.append(RussianNewsItem(
                            title=message.text[:100] + "..." if len(message.text) > 100 else message.text,
                            content=message.text,
                            published_at=message.date,
                            source=f"Telegram: {channel_username}",
                            url=f"https://t.me/{channel_username}/{message.id}",
                            symbol=self.extract_symbols(message.text)
                        ))
            
            logger.info(f"📱 Получено {len(news_items)} новостей из {channel_username}")
            
        except Exception as e:
            logger.error(f"Ошибка получения истории канала {channel_username}: {e}")
        
        return news_items
    
    def is_financial_news(self, text: str) -> bool:
        """Проверка, является ли текст финансовой новостью"""
        financial_keywords = [
            'акции', 'облигации', 'рубль', 'доллар', 'евро', 'нефть', 'газ',
            'биржа', 'торги', 'инвестиции', 'портфель', 'дивиденды', 'прибыль',
            'убыток', 'рост', 'падение', 'курс', 'валюта', 'фонд', 'индекс',
            'SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN', 'MGNT', 'MTSS',
            'MOEX', 'RTS', 'Мосбиржа', 'Газпром', 'Сбербанк', 'Лукойл'
        ]
        
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in financial_keywords)
    
    def extract_symbols(self, text: str) -> str:
        """Извлечение тикеров из текста"""
        symbols = []
        
        # Российские тикеры
        russian_tickers = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN', 'MGNT', 'MTSS', 'PIKK', 'IRAO', 'SGZH']
        
        for ticker in russian_tickers:
            if ticker in text.upper():
                symbols.append(ticker)
        
        return ', '.join(symbols) if symbols else 'GENERAL'
    
    async def get_all_channels_news(self, days_back: int = 7) -> Dict[str, List[RussianNewsItem]]:
        """Получение новостей из всех каналов"""
        all_news = {}
        
        for channel_name, channel_username in self.channels.items():
            try:
                news = await self.get_channel_history(channel_username, limit=50)
                
                # Фильтруем по времени
                cutoff_time = datetime.now() - timedelta(days=days_back)
                recent_news = [n for n in news if n.published_at >= cutoff_time]
                
                all_news[channel_name] = recent_news
                
                # Небольшая задержка между запросами
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Ошибка получения новостей из {channel_name}: {e}")
                all_news[channel_name] = []
        
        return all_news
    
    async def close(self):
        """Закрытие Telegram клиента"""
        if self.client:
            await self.client.disconnect()

class RussianSentimentAnalyzer:
    """Анализатор настроений для русского языка"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.financial_pipeline = None
        self.init_models()
    
    def init_models(self):
        """Инициализация моделей для русского языка"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers не доступен")
            return
        
        try:
            # Модель для анализа настроений на русском языке
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="blanchefort/rubert-base-cased-sentiment",
                return_all_scores=True
            )
            
            # Альтернативная модель для финансовых текстов
            try:
                self.financial_pipeline = pipeline(
                    "text-classification",
                    model="DeepPavlov/rubert-base-cased-sentiment",
                    return_all_scores=True
                )
            except:
                logger.warning("Финансовая модель недоступна, используем основную")
                self.financial_pipeline = self.sentiment_pipeline
            
            logger.info("✅ Модели анализа настроений для русского языка загружены")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            # Fallback на простой анализ ключевых слов
            self.sentiment_pipeline = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Анализ настроения русского текста"""
        if not text:
            return {'score': 0.0, 'confidence': 0.0}
        
        # Если модели недоступны, используем анализ ключевых слов
        if not self.sentiment_pipeline:
            return self.keyword_sentiment_analysis(text)
        
        try:
            # Ограничиваем длину текста
            text = text[:512]
            
            results = self.sentiment_pipeline(text)
            
            # Преобразуем результаты
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 0.0
            
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label or 'позитив' in label:
                    positive_score = score
                elif 'negative' in label or 'негатив' in label:
                    negative_score = score
                else:
                    neutral_score = score
            
            # Рассчитываем общий сентимент
            sentiment_score = positive_score - negative_score
            confidence = max(positive_score, negative_score, neutral_score)
            
            return {
                'score': sentiment_score,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа настроения: {e}")
            return self.keyword_sentiment_analysis(text)
    
    def keyword_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Анализ настроения по ключевым словам (fallback)"""
        positive_keywords = [
            'рост', 'повышение', 'увеличение', 'прибыль', 'успех', 'победа',
            'позитив', 'хорошо', 'отлично', 'прекрасно', 'замечательно',
            'стабильность', 'устойчивость', 'развитие', 'прогресс'
        ]
        
        negative_keywords = [
            'падение', 'снижение', 'уменьшение', 'убыток', 'провал', 'поражение',
            'негатив', 'плохо', 'ужасно', 'катастрофа', 'кризис',
            'нестабильность', 'риск', 'опасность', 'проблема'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
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

class RussianNewsAnalyzer:
    """Основной анализатор российских новостей"""
    
    def __init__(self, config_file: str = "russian_news_config.json"):
        self.config = self.load_config(config_file)
        self.sentiment_analyzer = RussianSentimentAnalyzer()
        self.moex_provider = MOEXNewsProvider()
        self.media_provider = RussianMediaProvider()
        self.telegram_provider = None
        
        # Инициализируем Telegram если настроен
        if self.config.get("telegram", {}).get("enabled", False):
            self.init_telegram()
        
        logger.info("✅ Анализатор российских новостей инициализирован")
    
    def load_config(self, config_file: str) -> Dict:
        """Загрузка конфигурации"""
        default_config = {
            "sources": {
                "moex": {"enabled": True},
                "russian_media": {"enabled": True},
                "telegram": {"enabled": False, "api_id": "", "api_hash": "", "phone": ""}
            },
            "telegram_channels": {
                "rbc_finance": "@rbc_finance",
                "vedomosti": "@vedomosti",
                "kommersant": "@kommersant",
                "tass_economics": "@tass_economics",
                "moex_official": "@moex_official",
                "investing_ru": "@investing_ru"
            },
            "cache": {
                "enabled": True,
                "ttl_hours": 2
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
    
    def init_telegram(self):
        """Инициализация Telegram провайдера"""
        telegram_config = self.config.get("telegram", {})
        
        if telegram_config.get("api_id") and telegram_config.get("api_hash"):
            self.telegram_provider = TelegramChannelProvider(
                api_id=telegram_config["api_id"],
                api_hash=telegram_config["api_hash"],
                phone=telegram_config.get("phone")
            )
        else:
            logger.warning("Telegram не настроен: отсутствуют API ключи")
    
    async def get_news_for_symbol(self, symbol: str, days_back: int = 7) -> List[RussianNewsItem]:
        """Получение новостей для символа из всех источников"""
        all_news = []
        
        # Получаем новости из MOEX
        if self.config["sources"]["moex"]["enabled"]:
            try:
                moex_news = await self.moex_provider.get_news(symbol, days_back)
                all_news.extend(moex_news)
                logger.info(f"📰 Получено {len(moex_news)} новостей из MOEX")
            except Exception as e:
                logger.error(f"Ошибка получения новостей MOEX: {e}")
        
        # Получаем новости из российских СМИ
        if self.config["sources"]["russian_media"]["enabled"]:
            try:
                media_news = await self.media_provider.get_news(symbol, days_back)
                all_news.extend(media_news)
                logger.info(f"📰 Получено {len(media_news)} новостей из российских СМИ")
            except Exception as e:
                logger.error(f"Ошибка получения новостей СМИ: {e}")
        
        # Получаем новости из Telegram каналов
        if self.config["sources"]["telegram"]["enabled"] and self.telegram_provider:
            try:
                if not self.telegram_provider.client:
                    await self.telegram_provider.initialize()
                
                telegram_news = await self.telegram_provider.get_all_channels_news(days_back)
                
                # Фильтруем новости по символу
                symbol_news = []
                for channel_news in telegram_news.values():
                    for news in channel_news:
                        if symbol in news.symbol or news.symbol == 'GENERAL':
                            symbol_news.append(news)
                
                all_news.extend(symbol_news)
                logger.info(f"📱 Получено {len(symbol_news)} новостей из Telegram")
                
            except Exception as e:
                logger.error(f"Ошибка получения новостей Telegram: {e}")
        
        # Анализируем настроения
        for news in all_news:
            if news.sentiment_score == 0.0:
                sentiment = self.sentiment_analyzer.analyze_sentiment(news.title + " " + news.content)
                news.sentiment_score = sentiment['score']
                news.confidence = sentiment['confidence']
        
        # Удаляем дубликаты
        unique_news = self.remove_duplicates(all_news)
        
        logger.info(f"📊 Итого получено {len(unique_news)} уникальных новостей для {symbol}")
        return unique_news
    
    def remove_duplicates(self, news_list: List[RussianNewsItem]) -> List[RussianNewsItem]:
        """Удаление дубликатов новостей"""
        unique_news = []
        seen_titles = set()
        
        for news in news_list:
            title_hash = hashlib.md5(news.title.lower().encode('utf-8')).hexdigest()
            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                unique_news.append(news)
        
        return unique_news
    
    def calculate_aggregate_sentiment(self, news_list: List[RussianNewsItem]) -> Dict[str, float]:
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
            hours_ago = (datetime.now() - news.published_at).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - hours_ago / (24 * 7))
            
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
        """Получение настроений рынка для списка российских символов"""
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
        if self.telegram_provider:
            await self.telegram_provider.close()

# Пример использования
async def main():
    """Пример использования анализатора российских новостей"""
    
    # Создаем анализатор
    analyzer = RussianNewsAnalyzer()
    
    # Российские символы для анализа
    russian_symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN']
    
    # Получаем настроения
    sentiments = await analyzer.get_market_sentiment(russian_symbols, days_back=3)
    
    # Выводим результаты
    print("\n📊 АНАЛИЗ НАСТРОЕНИЙ РОССИЙСКОГО РЫНКА")
    print("=" * 60)
    
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
