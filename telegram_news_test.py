#!/usr/bin/env python3
"""
Тестирование получения новостей из Telegram каналов
Скрипт для проверки работы с историей сообщений каналов
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

# Импортируем наш анализатор
from russian_news_analyzer import RussianNewsAnalyzer, TelegramChannelProvider

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramNewsTester:
    """Тестер для проверки работы с Telegram каналами"""
    
    def __init__(self, config_file: str = "russian_news_config.json"):
        self.config = self.load_config(config_file)
        self.telegram_provider = None
        self.results = {}
    
    def load_config(self, config_file: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return {}
    
    async def test_telegram_connection(self):
        """Тестирование подключения к Telegram"""
        telegram_config = self.config.get("telegram", {})
        
        if not telegram_config.get("enabled", False):
            logger.warning("Telegram отключен в конфигурации")
            return False
        
        api_id = telegram_config.get("api_id")
        api_hash = telegram_config.get("api_hash")
        phone = telegram_config.get("phone")
        
        if not api_id or not api_hash:
            logger.error("Отсутствуют API ключи для Telegram")
            return False
        
        try:
            self.telegram_provider = TelegramChannelProvider(api_id, api_hash, phone)
            success = await self.telegram_provider.initialize()
            
            if success:
                logger.info("✅ Подключение к Telegram успешно")
                return True
            else:
                logger.error("❌ Ошибка подключения к Telegram")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Telegram: {e}")
            return False
    
    async def test_single_channel(self, channel_username: str, limit: int = 20) -> Dict:
        """Тестирование одного канала"""
        if not self.telegram_provider:
            return {"error": "Telegram не инициализирован"}
        
        try:
            logger.info(f"🔍 Тестирование канала: {channel_username}")
            
            # Получаем историю сообщений
            news_items = await self.telegram_provider.get_channel_history(channel_username, limit)
            
            # Анализируем результаты
            total_messages = len(news_items)
            financial_news = [n for n in news_items if n.symbol != 'GENERAL']
            
            # Статистика по времени
            now = datetime.now()
            recent_news = [n for n in news_items if (now - n.published_at).days <= 7]
            
            result = {
                "channel": channel_username,
                "total_messages": total_messages,
                "financial_news": len(financial_news),
                "recent_news": len(recent_news),
                "success": True,
                "sample_titles": [n.title[:100] for n in news_items[:5]]
            }
            
            logger.info(f"📊 {channel_username}: {total_messages} сообщений, {len(financial_news)} финансовых")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования канала {channel_username}: {e}")
            return {
                "channel": channel_username,
                "error": str(e),
                "success": False
            }
    
    async def test_all_channels(self) -> Dict[str, Dict]:
        """Тестирование всех каналов из конфигурации"""
        channels = self.config.get("telegram_channels", {})
        results = {}
        
        for channel_name, channel_username in channels.items():
            result = await self.test_single_channel(channel_username, limit=10)
            results[channel_name] = result
            
            # Задержка между запросами
            await asyncio.sleep(2)
        
        return results
    
    async def test_historical_data(self, channel_username: str, days_back: int = 30) -> Dict:
        """Тестирование получения исторических данных"""
        if not self.telegram_provider:
            return {"error": "Telegram не инициализирован"}
        
        try:
            logger.info(f"📅 Получение исторических данных из {channel_username} за {days_back} дней")
            
            # Получаем больше сообщений для анализа истории
            news_items = await self.telegram_provider.get_channel_history(channel_username, limit=200)
            
            # Фильтруем по времени
            cutoff_date = datetime.now() - timedelta(days=days_back)
            historical_news = [n for n in news_items if n.published_at >= cutoff_date]
            
            # Группируем по дням
            daily_stats = {}
            for news in historical_news:
                date_key = news.published_at.strftime('%Y-%m-%d')
                if date_key not in daily_stats:
                    daily_stats[date_key] = 0
                daily_stats[date_key] += 1
            
            result = {
                "channel": channel_username,
                "total_historical": len(historical_news),
                "days_covered": len(daily_stats),
                "daily_stats": daily_stats,
                "date_range": {
                    "start": min(n.published_at for n in historical_news).strftime('%Y-%m-%d') if historical_news else None,
                    "end": max(n.published_at for n in historical_news).strftime('%Y-%m-%d') if historical_news else None
                }
            }
            
            logger.info(f"📊 Исторические данные: {len(historical_news)} новостей за {len(daily_stats)} дней")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения исторических данных: {e}")
            return {"error": str(e)}
    
    def save_results(self, results: Dict, filename: str = "telegram_test_results.json"):
        """Сохранение результатов тестирования"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"💾 Результаты сохранены в {filename}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов: {e}")
    
    def generate_report(self, results: Dict) -> str:
        """Генерация отчета по результатам тестирования"""
        report = []
        report.append("📊 ОТЧЕТ ПО ТЕСТИРОВАНИЮ TELEGRAM КАНАЛОВ")
        report.append("=" * 60)
        
        # Общая статистика
        total_channels = len(results)
        successful_channels = sum(1 for r in results.values() if r.get("success", False))
        
        report.append(f"Всего каналов: {total_channels}")
        report.append(f"Успешно протестировано: {successful_channels}")
        report.append(f"Процент успеха: {successful_channels/total_channels*100:.1f}%")
        report.append("")
        
        # Детали по каналам
        for channel_name, result in results.items():
            report.append(f"📱 {channel_name}:")
            if result.get("success", False):
                report.append(f"  ✅ Статус: Успешно")
                report.append(f"  📰 Всего сообщений: {result.get('total_messages', 0)}")
                report.append(f"  💰 Финансовых новостей: {result.get('financial_news', 0)}")
                report.append(f"  📅 За последнюю неделю: {result.get('recent_news', 0)}")
            else:
                report.append(f"  ❌ Статус: Ошибка")
                report.append(f"  🔍 Ошибка: {result.get('error', 'Неизвестная ошибка')}")
            report.append("")
        
        return "\n".join(report)
    
    async def close(self):
        """Закрытие соединений"""
        if self.telegram_provider:
            await self.telegram_provider.close()

async def main():
    """Основная функция тестирования"""
    
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ TELEGRAM КАНАЛОВ")
    print("=" * 50)
    
    # Создаем тестер
    tester = TelegramNewsTester()
    
    try:
        # Тестируем подключение
        print("1. Тестирование подключения к Telegram...")
        connection_ok = await tester.test_telegram_connection()
        
        if not connection_ok:
            print("❌ Не удалось подключиться к Telegram")
            print("💡 Проверьте настройки в russian_news_config.json")
            return
        
        # Тестируем все каналы
        print("\n2. Тестирование всех каналов...")
        results = await tester.test_all_channels()
        
        # Генерируем отчет
        report = tester.generate_report(results)
        print("\n" + report)
        
        # Сохраняем результаты
        tester.save_results(results)
        
        # Тестируем исторические данные для одного канала
        print("\n3. Тестирование исторических данных...")
        test_channel = "@rbc_finance"  # Можно изменить на любой канал
        historical_result = await tester.test_historical_data(test_channel, days_back=7)
        
        if "error" not in historical_result:
            print(f"📅 Исторические данные из {test_channel}:")
            print(f"  Всего новостей: {historical_result['total_historical']}")
            print(f"  Дней покрыто: {historical_result['days_covered']}")
            print(f"  Период: {historical_result['date_range']['start']} - {historical_result['date_range']['end']}")
        
        print("\n✅ Тестирование завершено!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")
    
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())
