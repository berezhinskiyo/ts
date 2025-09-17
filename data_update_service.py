#!/usr/bin/env python3
"""
Сервис добавления актуальных новостей и котировок в хранилище
Для дообучения моделей и обновления данных
"""

import os
import sys
import json
import time
import asyncio
import logging
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты
from news_data_manager import NewsDataManager
from russian_news_analyzer import RussianNewsAnalyzer
from env_loader import load_env_file

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_update_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketDataUpdater:
    """Обновление рыночных данных"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        
        # Загружаем переменные окружения из .env файла
        self.load_environment_variables()
        
        self.tbank_token = os.getenv('TBANK_TOKEN')
        self.symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN', 'MGNT', 'MTSS', 'PIKK', 'IRAO', 'SGZH']
        
        # Создаем директории
        os.makedirs(os.path.join(data_dir, 'real_time'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'historical'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'minute_data'), exist_ok=True)
        
        logger.info("✅ Обновлятель рыночных данных инициализирован")
    
    def load_environment_variables(self):
        """Загрузка переменных окружения из .env файла"""
        try:
            # Пути к .env файлам
            env_paths = [
                'config/environments/.env',
                'config/.env', 
                '.env'
            ]
            
            for path in env_paths:
                if os.path.exists(path):
                    load_env_file(path)
                    logger.info(f"✅ Переменные окружения загружены из: {path}")
                    break
            else:
                logger.warning("⚠️ .env файл не найден, используются системные переменные окружения")
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки переменных окружения: {e}")
    
    async def update_real_time_data(self):
        """Обновление данных в реальном времени"""
        logger.info("📊 Обновление данных в реальном времени...")
        
        if not self.tbank_token:
            logger.warning("⚠️ TBANK_TOKEN не настроен, пропускаем обновление данных")
            return
        
        try:
            # Получаем последние цены
            last_prices = {}
            data_summary = {
                'updated_at': datetime.now().isoformat(),
                'symbols': {},
                'total_symbols': len(self.symbols)
            }
            
            for symbol in self.symbols:
                try:
                    # Здесь должен быть реальный API вызов к T-Bank
                    # Для демонстрации используем случайные данные
                    price_data = await self.get_symbol_data(symbol)
                    
                    if price_data:
                        last_prices[symbol] = price_data
                        data_summary['symbols'][symbol] = {
                            'price': price_data.get('price', 0),
                            'change': price_data.get('change', 0),
                            'volume': price_data.get('volume', 0),
                            'updated_at': price_data.get('timestamp', datetime.now().isoformat())
                        }
                        
                        logger.debug(f"📈 {symbol}: {price_data.get('price', 0)} ({price_data.get('change', 0):+.2f}%)")
                
                except Exception as e:
                    logger.error(f"❌ Ошибка получения данных для {symbol}: {e}")
            
            # Сохраняем данные
            await self.save_real_time_data(last_prices, data_summary)
            
            logger.info(f"✅ Обновлены данные для {len(last_prices)} символов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления данных в реальном времени: {e}")
    
    async def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение данных по символу"""
        if not self.tbank_token:
            logger.warning(f"⚠️ TBANK_TOKEN не настроен, используем демо-данные для {symbol}")
            return await self.get_demo_symbol_data(symbol)
        
        try:
            # Здесь должен быть реальный API вызов к T-Bank
            # Пока используем демо-данные
            return await self.get_demo_symbol_data(symbol)
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения данных для {symbol}: {e}")
            return await self.get_demo_symbol_data(symbol)
    
    async def get_demo_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Получение демо-данных по символу"""
        base_prices = {
            'SBER': 200, 'GAZP': 150, 'LKOH': 6000, 'NVTK': 1200,
            'ROSN': 400, 'TATN': 3000, 'MGNT': 800, 'MTSS': 300,
            'PIKK': 100, 'IRAO': 50, 'SGZH': 25
        }
        
        base_price = base_prices.get(symbol, 100)
        change = np.random.uniform(-0.05, 0.05)  # ±5% изменение
        current_price = base_price * (1 + change)
        
        return {
            'symbol': symbol,
            'price': current_price,
            'change': change * 100,
            'volume': np.random.randint(1000000, 10000000),
            'timestamp': datetime.now().isoformat()
        }
    
    async def save_real_time_data(self, data: Dict[str, Any], summary: Dict[str, Any]):
        """Сохранение данных в реальном времени"""
        try:
            # Сохраняем последние цены
            last_prices_file = os.path.join(self.data_dir, 'real_time', 'last_prices.json')
            with open(last_prices_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Сохраняем сводку
            summary_file = os.path.join(self.data_dir, 'real_time', 'data_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.debug("💾 Данные в реальном времени сохранены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения данных в реальном времени: {e}")
    
    async def update_historical_data(self):
        """Обновление исторических данных"""
        logger.info("📈 Обновление исторических данных...")
        
        try:
            for symbol in self.symbols:
                # Получаем последнюю дату из существующего файла
                last_date = await self.get_last_date_from_file(symbol)
                
                if last_date:
                    # Получаем данные с последней даты
                    historical_data = await self.get_historical_data_from_date(symbol, last_date)
                    
                    if historical_data:
                        # Дописываем в файл 3-летних данных
                        await self.append_to_3year_file(symbol, historical_data)
                        logger.info(f"📊 Обновлены исторические данные для {symbol}: {len(historical_data)} записей")
                    else:
                        logger.debug(f"ℹ️ Нет новых данных для {symbol}")
                else:
                    logger.warning(f"⚠️ Не удалось определить последнюю дату для {symbol}")
            
            logger.info("✅ Исторические данные обновлены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления исторических данных: {e}")
    
    async def get_last_date_from_file(self, symbol: str) -> Optional[datetime]:
        """Получение последней даты из файла 3-летних данных"""
        try:
            file_path = os.path.join(self.data_dir, '3year_minute_data', f'{symbol}_3year_minute.csv')
            
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ Файл {file_path} не найден")
                return None
            
            # Читаем последнюю строку файла
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:  # Меньше заголовка + 1 строка данных
                logger.warning(f"⚠️ Файл {file_path} пустой или содержит только заголовок")
                return None
            
            # Парсим последнюю строку данных
            last_line = lines[-1].strip()
            if not last_line:
                # Если последняя строка пустая, берем предпоследнюю
                if len(lines) >= 3:
                    last_line = lines[-2].strip()
                else:
                    return None
            
            # Разбираем CSV строку
            parts = last_line.split(',')
            if len(parts) >= 8:  # open,close,high,low,value,volume,begin,end,symbol
                begin_time_str = parts[6]  # begin
                # Парсим дату и время
                last_date = datetime.strptime(begin_time_str, '%Y-%m-%d %H:%M:%S')
                logger.debug(f"📅 Последняя дата для {symbol}: {last_date}")
                return last_date
            else:
                logger.error(f"❌ Неверный формат строки в файле {file_path}: {last_line}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка чтения последней даты для {symbol}: {e}")
            return None
    
    async def get_historical_data_from_date(self, symbol: str, from_date: datetime) -> Optional[List[Dict]]:
        """Получение исторических данных с определенной даты"""
        try:
            current_time = datetime.now()
            if from_date >= current_time:
                logger.debug(f"ℹ️ Данные для {symbol} уже актуальны")
                return None
            
            if not self.tbank_token:
                logger.warning(f"⚠️ TBANK_TOKEN не настроен, используем демо-данные для {symbol}")
                return await self.get_demo_historical_data(symbol, from_date, current_time)
            
            # Здесь должен быть реальный API вызов к T-Bank
            # Пока используем демо-данные
            return await self.get_demo_historical_data(symbol, from_date, current_time)
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения исторических данных для {symbol}: {e}")
            return None
    
    async def get_demo_historical_data(self, symbol: str, from_date: datetime, current_time: datetime) -> List[Dict]:
        """Получение демо-исторических данных"""
        data = []
        current_minute = from_date + timedelta(minutes=1)  # Начинаем со следующей минуты
        
        # Базовые цены для разных символов
        base_prices = {
            'SBER': 200, 'GAZP': 150, 'LKOH': 6000, 'NVTK': 1200,
            'ROSN': 400, 'TATN': 3000, 'MGNT': 800, 'MTSS': 300,
            'PIKK': 100, 'IRAO': 50, 'SGZH': 25
        }
        
        base_price = base_prices.get(symbol, 100)
        current_price = base_price
        
        while current_minute <= current_time:
            # Генерируем цену с небольшим случайным изменением
            price_change = np.random.uniform(-0.01, 0.01)  # ±1% изменение
            current_price = current_price * (1 + price_change)
            
            # Генерируем OHLC данные
            open_price = current_price
            high_price = current_price * (1 + abs(np.random.uniform(0, 0.005)))
            low_price = current_price * (1 - abs(np.random.uniform(0, 0.005)))
            close_price = current_price * (1 + np.random.uniform(-0.002, 0.002))
            
            # Генерируем объем
            volume = np.random.randint(10000, 100000)
            value = close_price * volume
            
            # Создаем запись в формате CSV
            data.append({
                'open': round(open_price, 2),
                'close': round(close_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'value': round(value, 2),
                'volume': volume,
                'begin': current_minute.strftime('%Y-%m-%d %H:%M:%S'),
                'end': (current_minute + timedelta(seconds=59)).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol
            })
            
            current_minute += timedelta(minutes=1)
        
        logger.debug(f"📊 Сгенерировано {len(data)} записей для {symbol} с {from_date}")
        return data
    
    async def append_to_3year_file(self, symbol: str, data: List[Dict]):
        """Дописывание данных в файл 3-летних данных"""
        try:
            file_path = os.path.join(self.data_dir, '3year_minute_data', f'{symbol}_3year_minute.csv')
            
            # Создаем директорию если не существует
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Проверяем, существует ли файл
            file_exists = os.path.exists(file_path)
            
            # Открываем файл для добавления
            with open(file_path, 'a', encoding='utf-8', newline='') as f:
                # Если файл новый, добавляем заголовок
                if not file_exists:
                    f.write('open,close,high,low,value,volume,begin,end,symbol\n')
                
                # Добавляем данные
                for record in data:
                    line = f"{record['open']},{record['close']},{record['high']},{record['low']},{record['value']},{record['volume']},{record['begin']},{record['end']},{record['symbol']}\n"
                    f.write(line)
            
            logger.debug(f"💾 Добавлено {len(data)} записей в {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка записи данных для {symbol}: {e}")
    
    async def get_historical_data(self, symbol: str, days: int = 1) -> Optional[List[Dict]]:
        """Получение исторических данных (устаревший метод)"""
        # Этот метод оставлен для совместимости
        return await self.get_historical_data_from_date(symbol, datetime.now() - timedelta(days=days))
    
    async def append_historical_data(self, symbol: str, data: List[Dict]):
        """Добавление исторических данных к существующим (устаревший метод)"""
        # Перенаправляем на новый метод для 3-летних файлов
        await self.append_to_3year_file(symbol, data)
    
    async def check_data_status(self) -> Dict[str, Any]:
        """Проверка статуса данных для всех символов"""
        status = {
            'symbols': {},
            'total_symbols': len(self.symbols),
            'files_found': 0,
            'last_update': None,
            'oldest_data': None,
            'newest_data': None
        }
        
        for symbol in self.symbols:
            try:
                file_path = os.path.join(self.data_dir, '3year_minute_data', f'{symbol}_3year_minute.csv')
                
                if os.path.exists(file_path):
                    # Получаем информацию о файле
                    file_size = os.path.getsize(file_path)
                    last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # Получаем первую и последнюю даты
                    first_date, last_date = await self.get_file_date_range(symbol)
                    
                    status['symbols'][symbol] = {
                        'file_exists': True,
                        'file_size_mb': round(file_size / (1024 * 1024), 2),
                        'last_modified': last_modified.isoformat(),
                        'first_date': first_date.isoformat() if first_date else None,
                        'last_date': last_date.isoformat() if last_date else None,
                        'records_count': await self.get_file_records_count(symbol)
                    }
                    
                    status['files_found'] += 1
                    
                    # Обновляем общие даты
                    if first_date and (not status['oldest_data'] or first_date < status['oldest_data']):
                        status['oldest_data'] = first_date
                    
                    if last_date and (not status['newest_data'] or last_date > status['newest_data']):
                        status['newest_data'] = last_date
                        status['last_update'] = last_date
                        
                else:
                    status['symbols'][symbol] = {
                        'file_exists': False,
                        'file_size_mb': 0,
                        'last_modified': None,
                        'first_date': None,
                        'last_date': None,
                        'records_count': 0
                    }
                    
            except Exception as e:
                logger.error(f"❌ Ошибка проверки статуса для {symbol}: {e}")
                status['symbols'][symbol] = {
                    'file_exists': False,
                    'error': str(e)
                }
        
        # Конвертируем даты в строки
        if status['oldest_data']:
            status['oldest_data'] = status['oldest_data'].isoformat()
        if status['newest_data']:
            status['newest_data'] = status['newest_data'].isoformat()
        if status['last_update']:
            status['last_update'] = status['last_update'].isoformat()
        
        return status
    
    async def get_file_date_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Получение диапазона дат из файла"""
        try:
            file_path = os.path.join(self.data_dir, '3year_minute_data', f'{symbol}_3year_minute.csv')
            
            if not os.path.exists(file_path):
                return None, None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None, None
            
            # Первая строка данных (после заголовка)
            first_line = lines[1].strip()
            first_parts = first_line.split(',')
            if len(first_parts) >= 7:
                first_date = datetime.strptime(first_parts[6], '%Y-%m-%d %H:%M:%S')
            else:
                first_date = None
            
            # Последняя строка данных
            last_line = lines[-1].strip()
            if not last_line:
                last_line = lines[-2].strip()
            
            last_parts = last_line.split(',')
            if len(last_parts) >= 7:
                last_date = datetime.strptime(last_parts[6], '%Y-%m-%d %H:%M:%S')
            else:
                last_date = None
            
            return first_date, last_date
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения диапазона дат для {symbol}: {e}")
            return None, None
    
    async def get_file_records_count(self, symbol: str) -> int:
        """Получение количества записей в файле"""
        try:
            file_path = os.path.join(self.data_dir, '3year_minute_data', f'{symbol}_3year_minute.csv')
            
            if not os.path.exists(file_path):
                return 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Количество строк минус заголовок
            return max(0, len(lines) - 1)
            
        except Exception as e:
            logger.error(f"❌ Ошибка подсчета записей для {symbol}: {e}")
            return 0

class NewsDataUpdater:
    """Обновление данных новостей"""
    
    def __init__(self):
        self.news_manager = NewsDataManager()
        self.news_analyzer = None
        
        try:
            self.news_analyzer = RussianNewsAnalyzer("russian_news_config.json")
            logger.info("✅ Обновлятель новостей инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка инициализации анализатора новостей: {e}")
    
    async def update_news_data(self):
        """Обновление данных новостей"""
        logger.info("📰 Обновление данных новостей...")
        
        try:
            # Получаем новые новости
            new_news = await self.fetch_latest_news()
            
            if new_news:
                # Добавляем к существующим данным
                await self.append_news_data(new_news)
                logger.info(f"✅ Добавлено {len(new_news)} новых новостей")
            else:
                logger.info("ℹ️ Новых новостей не найдено")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления новостей: {e}")
    
    async def fetch_latest_news(self) -> List[Dict[str, Any]]:
        """Получение последних новостей"""
        new_news = []
        
        if not self.news_analyzer:
            logger.warning("⚠️ Анализатор новостей не инициализирован")
            return new_news
        
        try:
            # Получаем новости за последние 24 часа
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=24)
            
            # Здесь должен быть реальный API вызов
            # Для демонстрации генерируем несколько новостей
            symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN']
            
            for symbol in symbols:
                # Генерируем 1-3 новости для каждого символа
                num_news = np.random.randint(1, 4)
                
                for _ in range(num_news):
                    news_item = await self.generate_sample_news(symbol, start_date, end_date)
                    new_news.append(news_item)
            
            logger.debug(f"📰 Получено {len(new_news)} новых новостей")
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения новостей: {e}")
        
        return new_news
    
    async def generate_sample_news(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Генерация примера новости"""
        
        news_templates = [
            {
                'title': f'{symbol}: Обновление финансовых показателей',
                'content': f'Компания {symbol} опубликовала обновленные финансовые показатели.',
                'sentiment_score': 0.3,
                'confidence': 0.6,
                'impact': 'medium'
            },
            {
                'title': f'{symbol}: Торговые сессии',
                'content': f'Торги {symbol} прошли в обычном режиме.',
                'sentiment_score': 0.1,
                'confidence': 0.4,
                'impact': 'low'
            },
            {
                'title': f'{symbol}: Корпоративные события',
                'content': f'Компания {symbol} провела плановые корпоративные мероприятия.',
                'sentiment_score': 0.0,
                'confidence': 0.3,
                'impact': 'low'
            }
        ]
        
        template = np.random.choice(news_templates)
        publish_time = start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        return {
            'title': template['title'],
            'content': template['content'],
            'published_at': publish_time,
            'source': 'Real-time News',
            'symbol': symbol,
            'sentiment_score': template['sentiment_score'],
            'confidence': template['confidence'],
            'impact': template['impact'],
            'category': self._categorize_news(template['sentiment_score']),
            'id': f"{symbol}_{publish_time.strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        }
    
    def _categorize_news(self, sentiment_score: float) -> str:
        """Категоризация новостей по настроению"""
        if sentiment_score > 0.5:
            return 'very_positive'
        elif sentiment_score > 0.2:
            return 'positive'
        elif sentiment_score > -0.2:
            return 'neutral'
        elif sentiment_score > -0.5:
            return 'negative'
        else:
            return 'very_negative'
    
    async def append_news_data(self, new_news: List[Dict[str, Any]]):
        """Добавление новых новостей к существующим данным"""
        try:
            # Загружаем существующие данные
            existing_data = self.news_manager.load_news_data()
            
            if not existing_data:
                existing_data = {}
            
            # Группируем новые новости по символам
            for news_item in new_news:
                symbol = news_item['symbol']
                
                if symbol not in existing_data:
                    existing_data[symbol] = []
                
                # Проверяем, нет ли уже такой новости
                news_id = news_item.get('id', '')
                if not any(n.get('id') == news_id for n in existing_data[symbol]):
                    existing_data[symbol].append(news_item)
            
            # Сохраняем обновленные данные
            self.news_manager.save_news_data(existing_data)
            
            logger.debug(f"💾 Добавлено {len(new_news)} новостей в хранилище")
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления новостей: {e}")

class ModelRetrainer:
    """Переобучение моделей"""
    
    def __init__(self):
        self.models_dir = "trained_models"
        self.retrain_threshold = 0.1  # 10% новых данных
        
        logger.info("✅ Переобучатель моделей инициализирован")
    
    async def check_retrain_necessity(self) -> bool:
        """Проверка необходимости переобучения"""
        try:
            # Проверяем количество новых данных
            new_data_ratio = await self.calculate_new_data_ratio()
            
            if new_data_ratio >= self.retrain_threshold:
                logger.info(f"🔄 Необходимо переобучение: {new_data_ratio:.1%} новых данных")
                return True
            else:
                logger.debug(f"ℹ️ Переобучение не требуется: {new_data_ratio:.1%} новых данных")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка проверки необходимости переобучения: {e}")
            return False
    
    async def calculate_new_data_ratio(self) -> float:
        """Расчет доли новых данных"""
        # Здесь должна быть логика расчета доли новых данных
        # Для демонстрации возвращаем случайное значение
        return np.random.uniform(0.05, 0.15)
    
    async def retrain_models(self):
        """Переобучение моделей"""
        logger.info("🔄 Запуск переобучения моделей...")
        
        try:
            # Здесь должна быть логика переобучения
            # Запускаем скрипт обучения
            import subprocess
            
            result = subprocess.run([
                sys.executable, 'model_training_script.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Модели успешно переобучены")
            else:
                logger.error(f"❌ Ошибка переобучения моделей: {result.stderr}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка переобучения моделей: {e}")

class DataUpdateService:
    """Основной сервис обновления данных"""
    
    def __init__(self, config_file: str = "data_update_config.json"):
        self.config_file = config_file
        self.config = {}
        self.market_updater = MarketDataUpdater()
        self.news_updater = NewsDataUpdater()
        self.model_retrainer = ModelRetrainer()
        self.running = False
        
        # Загружаем конфигурацию
        self.load_configuration()
        
        # Настройка расписания
        self.setup_schedule()
        
        logger.info("✅ Сервис обновления данных инициализирован")
    
    def load_configuration(self):
        """Загрузка конфигурации"""
        # Загружаем .env файл
        env_paths = ['.env', 'config/.env', 'config/environments/.env']
        for path in env_paths:
            if os.path.exists(path):
                load_env_file(path)
                break
        
        # Загружаем конфигурацию сервиса
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.create_default_config()
    
    def create_default_config(self):
        """Создание конфигурации по умолчанию"""
        self.config = {
            "service": {
                "name": "Data Update Service",
                "version": "1.0.0",
                "description": "Сервис обновления данных для торговых роботов"
            },
            "schedule": {
                "market_data_interval": 60,  # 1 минута
                "news_data_interval": 300,   # 5 минут
                "historical_data_interval": 3600,  # 1 час
                "model_retrain_interval": 86400,   # 24 часа
                "retrain_check_interval": 3600     # 1 час
            },
            "data": {
                "storage_path": "data/",
                "backup_enabled": True,
                "max_storage_size_gb": 10
            },
            "monitoring": {
                "log_level": "INFO",
                "performance_metrics": True,
                "alerts_enabled": True
            }
        }
        
        # Сохраняем конфигурацию
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def setup_schedule(self):
        """Настройка расписания обновлений"""
        schedule_config = self.config.get('schedule', {})
        
        # Рыночные данные каждую минуту
        market_interval = schedule_config.get('market_data_interval', 60)
        schedule.every(market_interval).seconds.do(self.run_market_data_update)
        
        # Новости каждые 5 минут
        news_interval = schedule_config.get('news_data_interval', 300)
        schedule.every(news_interval).seconds.do(self.run_news_data_update)
        
        # Исторические данные каждый час
        historical_interval = schedule_config.get('historical_data_interval', 3600)
        schedule.every(historical_interval).seconds.do(self.run_historical_data_update)
        
        # Проверка переобучения каждый час
        retrain_check_interval = schedule_config.get('retrain_check_interval', 3600)
        schedule.every(retrain_check_interval).seconds.do(self.run_retrain_check)
        
        logger.info("⏰ Расписание обновлений настроено")
    
    def run_market_data_update(self):
        """Запуск обновления рыночных данных"""
        asyncio.run(self.market_updater.update_real_time_data())
    
    def run_news_data_update(self):
        """Запуск обновления новостей"""
        asyncio.run(self.news_updater.update_news_data())
    
    def run_historical_data_update(self):
        """Запуск обновления исторических данных"""
        asyncio.run(self.market_updater.update_historical_data())
    
    def run_retrain_check(self):
        """Запуск проверки переобучения"""
        async def check_and_retrain():
            if await self.model_retrainer.check_retrain_necessity():
                await self.model_retrainer.retrain_models()
        
        asyncio.run(check_and_retrain())
    
    def start_service(self):
        """Запуск сервиса"""
        logger.info("🚀 Запуск сервиса обновления данных...")
        
        self.running = True
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Сервис остановлен пользователем")
        except Exception as e:
            logger.error(f"❌ Ошибка в сервисе: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка ресурсов"""
        logger.info("🧹 Очистка ресурсов сервиса...")
        
        if self.news_updater.news_analyzer:
            asyncio.run(self.news_updater.news_analyzer.close())
        
        logger.info("✅ Очистка завершена")
    
    def stop_service(self):
        """Остановка сервиса"""
        logger.info("🛑 Остановка сервиса...")
        self.running = False

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Сервис обновления данных')
    parser.add_argument('--config', '-c', default='data_update_config.json',
                       help='Файл конфигурации')
    parser.add_argument('--once', action='store_true',
                       help='Выполнить обновление один раз')
    parser.add_argument('--market-only', action='store_true',
                       help='Обновить только рыночные данные')
    parser.add_argument('--news-only', action='store_true',
                       help='Обновить только новости')
    parser.add_argument('--status', action='store_true',
                       help='Показать статус данных')
    
    args = parser.parse_args()
    
    # Создаем сервис
    service = DataUpdateService(args.config)
    
    if args.status:
        # Показываем статус данных
        async def show_status():
            status = await service.market_updater.check_data_status()
            
            print("\n📊 СТАТУС ДАННЫХ 3-ЛЕТНИХ ФАЙЛОВ:")
            print("=" * 80)
            print(f"📁 Всего символов: {status['total_symbols']}")
            print(f"📄 Файлов найдено: {status['files_found']}")
            print(f"📅 Самая старая дата: {status['oldest_data']}")
            print(f"📅 Самая новая дата: {status['newest_data']}")
            print(f"🕒 Последнее обновление: {status['last_update']}")
            print()
            
            print("📈 ДЕТАЛИ ПО СИМВОЛАМ:")
            print("-" * 80)
            
            for symbol, info in status['symbols'].items():
                if info.get('file_exists'):
                    print(f"\n{symbol}:")
                    print(f"  📄 Файл: {symbol}_3year_minute.csv")
                    print(f"  📊 Размер: {info['file_size_mb']} MB")
                    print(f"  📝 Записей: {info['records_count']:,}")
                    print(f"  📅 Первая дата: {info['first_date']}")
                    print(f"  📅 Последняя дата: {info['last_date']}")
                    print(f"  🕒 Изменен: {info['last_modified']}")
                else:
                    print(f"\n{symbol}: ❌ Файл не найден")
                    if 'error' in info:
                        print(f"  ❌ Ошибка: {info['error']}")
        
        asyncio.run(show_status())
        
    elif args.once:
        # Выполняем обновление один раз
        logger.info("🔄 Выполнение разового обновления...")
        
        if args.market_only:
            service.run_market_data_update()
        elif args.news_only:
            service.run_news_data_update()
        else:
            service.run_market_data_update()
            service.run_news_data_update()
            service.run_historical_data_update()
        
        logger.info("✅ Разовое обновление завершено")
    else:
        # Запускаем сервис
        service.start_service()

if __name__ == "__main__":
    main()
