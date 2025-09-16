#!/usr/bin/env python3
"""
Тест системы уведомлений в Telegram
"""

import asyncio
import os
import sys
import argparse
from datetime import datetime
import logging

# Добавляем путь к модулям
sys.path.append('.')

from telegram_notifications import TradingNotifier, TradeNotification, PortfolioUpdate, AlertNotification

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_telegram_notifications():
    """Тест системы уведомлений"""
    
    logger.info("🤖 ТЕСТИРОВАНИЕ TELEGRAM УВЕДОМЛЕНИЙ")
    logger.info("=" * 50)
    
    # Проверяем настройки
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        logger.error("❌ TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не настроены в .env файле")
        logger.info("📝 Добавьте в .env файл:")
        logger.info("   TELEGRAM_BOT_TOKEN=your_bot_token")
        logger.info("   TELEGRAM_CHAT_ID=your_chat_id")
        return False
    
    logger.info("✅ Настройки Telegram найдены")
    
    # Создаем уведомлятель с токенами из переменных окружения
    notifier = TradingNotifier(bot_token=bot_token, chat_id=chat_id)
    
    try:
        # Тест 1: Уведомление о покупке
        logger.info("\n📊 Тест 1: Уведомление о покупке")
        await notifier.notify_trade(
            symbol="GAZP",
            action="BUY",
            quantity=100,
            price=150.50,
            strategy="ARIMA",
            confidence=0.85
        )
        
        await asyncio.sleep(1)
        
        # Тест 2: Уведомление о продаже с прибылью
        logger.info("\n💰 Тест 2: Уведомление о продаже с прибылью")
        await notifier.notify_trade(
            symbol="GAZP",
            action="SELL",
            quantity=100,
            price=152.30,
            strategy="ARIMA",
            pnl=180.0,
            reason="TAKE_PROFIT"
        )
        
        await asyncio.sleep(1)
        
        # Тест 3: Уведомление о стоп-лоссе
        logger.info("\n🛑 Тест 3: Уведомление о стоп-лоссе")
        await notifier.notify_trade(
            symbol="SBER",
            action="SELL",
            quantity=50,
            price=295.20,
            strategy="LSTM",
            pnl=-150.0,
            reason="STOP_LOSS"
        )
        
        await asyncio.sleep(1)
        
        # Тест 4: Обновление портфеля
        logger.info("\n📈 Тест 4: Обновление портфеля")
        await notifier.notify_portfolio_update(
            total_value=105000,
            total_pnl=5000,
            positions_count=3,
            daily_pnl=180.0
        )
        
        await asyncio.sleep(1)
        
        # Тест 5: Алерт о риске
        logger.info("\n⚠️ Тест 5: Алерт о риске")
        await notifier.notify_alert(
            alert_type="RISK",
            title="Превышен лимит риска",
            message="Текущий риск портфеля превышает 15%. Рекомендуется закрыть часть позиций.",
            severity="HIGH"
        )
        
        await asyncio.sleep(1)
        
        # Тест 6: Информационный алерт
        logger.info("\nℹ️ Тест 6: Информационный алерт")
        await notifier.notify_alert(
            alert_type="INFO",
            title="Завершение торговой сессии",
            message="Торговая сессия завершена. Все позиции закрыты. Дневная прибыль: +2.1%",
            severity="INFO"
        )
        
        await asyncio.sleep(1)
        
        # Тест 7: Дневная сводка
        logger.info("\n📊 Тест 7: Дневная сводка")
        await notifier.send_daily_summary()
        
        logger.info("\n✅ Все тесты уведомлений выполнены успешно!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании уведомлений: {e}")
        return False

async def test_rate_limiting():
    """Тест ограничений скорости отправки"""
    
    logger.info("\n🚦 ТЕСТИРОВАНИЕ ОГРАНИЧЕНИЙ СКОРОСТИ")
    logger.info("=" * 40)
    
    notifier = TradingNotifier(bot_token=os.getenv('TELEGRAM_BOT_TOKEN'), chat_id=os.getenv('TELEGRAM_CHAT_ID'))
    
    # Отправляем много уведомлений быстро
    for i in range(15):
        await notifier.notify_trade(
            symbol=f"TEST{i}",
            action="BUY",
            quantity=10,
            price=100.0,
            strategy="TEST"
        )
        await asyncio.sleep(0.1)  # Быстрая отправка
    
    logger.info("✅ Тест ограничений скорости завершен")

async def test_error_handling():
    """Тест обработки ошибок"""
    
    logger.info("\n🔧 ТЕСТИРОВАНИЕ ОБРАБОТКИ ОШИБОК")
    logger.info("=" * 40)
    
    # Создаем уведомлятель с неверными настройками
    notifier = TradingNotifier(bot_token="invalid_token", chat_id="invalid_chat")
    
    # Пытаемся отправить уведомление
    result = await notifier.notify_trade(
        symbol="TEST",
        action="BUY",
        quantity=10,
        price=100.0,
        strategy="TEST"
    )
    
    if not result:
        logger.info("✅ Обработка ошибок работает корректно")
    else:
        logger.warning("⚠️ Обработка ошибок может работать некорректно")

def print_setup_instructions():
    """Вывод инструкций по настройке"""
    
    print("\n" + "="*60)
    print("📱 НАСТРОЙКА TELEGRAM УВЕДОМЛЕНИЙ")
    print("="*60)
    print()
    print("1. Создайте бота в Telegram:")
    print("   - Напишите @BotFather в Telegram")
    print("   - Отправьте команду /newbot")
    print("   - Следуйте инструкциям для создания бота")
    print("   - Сохраните полученный токен")
    print()
    print("2. Получите Chat ID:")
    print("   - Напишите вашему боту любое сообщение")
    print("   - Перейдите по ссылке: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates")
    print("   - Найдите 'chat':{'id': <YOUR_CHAT_ID>}")
    print()
    print("3. Добавьте в .env файл:")
    print("   TELEGRAM_BOT_TOKEN=your_bot_token_here")
    print("   TELEGRAM_CHAT_ID=your_chat_id_here")
    print()
    print("4. Запустите тест:")
    print("   python test_telegram_notifications.py")
    print("   # Или с указанием пути к .env файлу:")
    print("   python test_telegram_notifications.py --env-file config/environments/.env")
    print("   # Или с прямым указанием токенов:")
    print("   python test_telegram_notifications.py --bot-token YOUR_TOKEN --chat-id YOUR_CHAT_ID")
    print()

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Тест системы Telegram уведомлений')
    parser.add_argument(
        '--env-file', 
        type=str, 
        default='.env',
        help='Путь к .env файлу (по умолчанию: .env)'
    )
    parser.add_argument(
        '--bot-token',
        type=str,
        help='Telegram bot token (переопределяет .env файл)'
    )
    parser.add_argument(
        '--chat-id',
        type=str,
        help='Telegram chat ID (переопределяет .env файл)'
    )
    return parser.parse_args()

async def main():
    """Основная функция"""
    
    # Парсим аргументы
    args = parse_arguments()
    
    print_setup_instructions()
    
    # Проверяем наличие .env файла
    if not os.path.exists(args.env_file):
        logger.error(f"❌ Файл {args.env_file} не найден")
        logger.info("📝 Создайте файл .env с настройками Telegram или укажите путь через --env-file")
        return
    
    # Загружаем переменные окружения
    try:
        from env_loader import load_env_file
        success = load_env_file(args.env_file, verbose=True)
        if not success:
            logger.error("❌ Не удалось загрузить .env файл")
            return
    except ImportError:
        # Fallback к стандартному способу
        try:
            from dotenv import load_dotenv
            load_dotenv(args.env_file)
            logger.info(f"✅ Переменные окружения загружены из {args.env_file}")
        except ImportError:
            logger.warning("⚠️ python-dotenv не установлен. Установите: pip install python-dotenv")
            return
    
    # Переопределяем переменные из аргументов командной строки
    if args.bot_token:
        os.environ['TELEGRAM_BOT_TOKEN'] = args.bot_token
        logger.info("✅ Bot token переопределен из аргументов")
    
    if args.chat_id:
        os.environ['TELEGRAM_CHAT_ID'] = args.chat_id
        logger.info("✅ Chat ID переопределен из аргументов")
    
    # Запускаем тесты
    success = await test_telegram_notifications()
    
    if success:
        await test_rate_limiting()
        await test_error_handling()
        
        logger.info("\n🎉 ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
        logger.info("📱 Telegram уведомления готовы к использованию")
    else:
        logger.error("\n❌ ТЕСТЫ НЕ ПРОШЛИ")
        logger.info("🔧 Проверьте настройки Telegram в .env файле")

if __name__ == "__main__":
    asyncio.run(main())
