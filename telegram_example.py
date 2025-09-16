#!/usr/bin/env python3
"""
Пример использования системы Telegram уведомлений
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

from telegram_notifications import TradingNotifier

async def example_trading_session():
    """Пример торговой сессии с уведомлениями"""
    
    print("🚀 ЗАПУСК ПРИМЕРА ТОРГОВОЙ СЕССИИ")
    print("=" * 50)
    
    # Создаем уведомлятель
    notifier = TradingNotifier()
    
    # Симулируем торговую сессию
    print("\n📊 Начало торговой сессии...")
    
    # Уведомление о начале сессии
    await notifier.notify_alert(
        alert_type="INFO",
        title="Начало торговой сессии",
        message="Торговая сессия началась. Система готова к работе.",
        severity="INFO"
    )
    
    await asyncio.sleep(2)
    
    # Симулируем несколько сделок
    trades = [
        {"symbol": "GAZP", "action": "BUY", "quantity": 100, "price": 150.50, "strategy": "ARIMA", "confidence": 0.85},
        {"symbol": "SBER", "action": "BUY", "quantity": 50, "price": 300.20, "strategy": "LSTM", "confidence": 0.78},
        {"symbol": "GAZP", "action": "SELL", "quantity": 100, "price": 152.30, "strategy": "ARIMA", "pnl": 180.0, "reason": "TAKE_PROFIT"},
        {"symbol": "PIKK", "action": "BUY", "quantity": 30, "price": 500.00, "strategy": "SARIMA", "confidence": 0.72},
        {"symbol": "SBER", "action": "SELL", "quantity": 50, "price": 295.80, "strategy": "LSTM", "pnl": -220.0, "reason": "STOP_LOSS"},
    ]
    
    for i, trade in enumerate(trades, 1):
        print(f"\n📈 Сделка {i}: {trade['action']} {trade['symbol']}")
        
        await notifier.notify_trade(
            symbol=trade["symbol"],
            action=trade["action"],
            quantity=trade["quantity"],
            price=trade["price"],
            strategy=trade["strategy"],
            pnl=trade.get("pnl"),
            reason=trade.get("reason"),
            confidence=trade.get("confidence")
        )
        
        await asyncio.sleep(1)
    
    # Обновление портфеля
    print("\n📊 Обновление портфеля...")
    await notifier.notify_portfolio_update(
        total_value=102500,
        total_pnl=2500,
        positions_count=2,
        daily_pnl=-40.0
    )
    
    await asyncio.sleep(2)
    
    # Алерт о риске
    print("\n⚠️ Алерт о риске...")
    await notifier.notify_alert(
        alert_type="RISK",
        title="Высокая волатильность",
        message="Обнаружена высокая волатильность на рынке. Рекомендуется снизить размеры позиций.",
        severity="MEDIUM"
    )
    
    await asyncio.sleep(2)
    
    # Дневная сводка
    print("\n📈 Дневная сводка...")
    await notifier.send_daily_summary()
    
    print("\n✅ Торговая сессия завершена!")

async def example_portfolio_monitoring():
    """Пример мониторинга портфеля"""
    
    print("\n📊 ПРИМЕР МОНИТОРИНГА ПОРТФЕЛЯ")
    print("=" * 40)
    
    notifier = TradingNotifier()
    
    # Симулируем изменения портфеля в течение дня
    portfolio_updates = [
        {"value": 100000, "pnl": 0, "positions": 0, "daily_pnl": 0},
        {"value": 100500, "pnl": 500, "positions": 1, "daily_pnl": 500},
        {"value": 101200, "pnl": 1200, "positions": 2, "daily_pnl": 1200},
        {"value": 100800, "pnl": 800, "positions": 2, "daily_pnl": 800},
        {"value": 102000, "pnl": 2000, "positions": 3, "daily_pnl": 2000},
    ]
    
    for i, update in enumerate(portfolio_updates, 1):
        print(f"\n📈 Обновление {i}: {update['value']:,.0f}₽ (P&L: {update['pnl']:+,.0f}₽)")
        
        await notifier.notify_portfolio_update(
            total_value=update["value"],
            total_pnl=update["pnl"],
            positions_count=update["positions"],
            daily_pnl=update["daily_pnl"]
        )
        
        await asyncio.sleep(1)
    
    print("\n✅ Мониторинг портфеля завершен!")

async def example_risk_alerts():
    """Пример алертов о рисках"""
    
    print("\n⚠️ ПРИМЕР АЛЕРТОВ О РИСКАХ")
    print("=" * 35)
    
    notifier = TradingNotifier()
    
    # Различные типы алертов
    alerts = [
        {
            "type": "RISK",
            "title": "Превышен лимит риска",
            "message": "Текущий риск портфеля превышает 15%. Рекомендуется закрыть часть позиций.",
            "severity": "HIGH"
        },
        {
            "type": "PERFORMANCE",
            "title": "Достигнута цель доходности",
            "message": "Портфель достиг месячной цели доходности 5%. Рассмотрите возможность фиксации прибыли.",
            "severity": "MEDIUM"
        },
        {
            "type": "ERROR",
            "title": "Ошибка подключения к API",
            "message": "Не удается подключиться к T-Bank API. Проверьте интернет-соединение и токены.",
            "severity": "CRITICAL"
        },
        {
            "type": "INFO",
            "title": "Завершение торговой сессии",
            "message": "Торговая сессия завершена. Все позиции закрыты. Дневная прибыль: +2.1%",
            "severity": "INFO"
        }
    ]
    
    for i, alert in enumerate(alerts, 1):
        print(f"\n🚨 Алерт {i}: {alert['title']}")
        
        await notifier.notify_alert(
            alert_type=alert["type"],
            title=alert["title"],
            message=alert["message"],
            severity=alert["severity"]
        )
        
        await asyncio.sleep(1)
    
    print("\n✅ Алерты отправлены!")

async def main():
    """Основная функция"""
    
    # Проверяем настройки
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не настроены")
        print("📝 Добавьте в .env файл:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")
        return
    
    print("✅ Настройки Telegram найдены")
    
    # Запускаем примеры
    await example_trading_session()
    await example_portfolio_monitoring()
    await example_risk_alerts()
    
    print("\n🎉 ВСЕ ПРИМЕРЫ ЗАВЕРШЕНЫ!")
    print("📱 Проверьте Telegram для просмотра уведомлений")

if __name__ == "__main__":
    asyncio.run(main())

