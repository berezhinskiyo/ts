#!/usr/bin/env python3
"""
Система уведомлений в Telegram для торговых сделок
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os
try:
    from config.parameters.config import Config
except ImportError:
    # Fallback для случаев, когда config не найден
    class Config:
        TELEGRAM_BOT_TOKEN = None
        TELEGRAM_CHAT_ID = None

# Дополнительная загрузка переменных окружения из os.environ
# Это нужно для случаев, когда .env файл загружен через env_loader
import os
if not Config.TELEGRAM_BOT_TOKEN:
    Config.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not Config.TELEGRAM_CHAT_ID:
    Config.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeNotification:
    """Уведомление о сделке"""
    symbol: str
    action: str  # 'BUY' или 'SELL'
    quantity: int
    price: float
    amount: float
    strategy: str
    timestamp: str
    pnl: Optional[float] = None
    reason: Optional[str] = None  # 'STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL'
    confidence: Optional[float] = None

@dataclass
class PortfolioUpdate:
    """Обновление портфеля"""
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    positions_count: int
    timestamp: str
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0

@dataclass
class AlertNotification:
    """Алерт-уведомление"""
    alert_type: str  # 'RISK', 'PERFORMANCE', 'ERROR', 'INFO'
    title: str
    message: str
    timestamp: str
    severity: str = 'INFO'  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

class TelegramNotifier:
    """Класс для отправки уведомлений в Telegram"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or Config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.session = None
        
        # Настройки форматирования
        self.emoji_map = {
            'BUY': '🟢',
            'SELL': '🔴',
            'STOP_LOSS': '🛑',
            'TAKE_PROFIT': '💰',
            'SIGNAL': '📊',
            'RISK': '⚠️',
            'PERFORMANCE': '📈',
            'ERROR': '❌',
            'INFO': 'ℹ️',
            'SUCCESS': '✅',
            'WARNING': '⚠️',
            'CRITICAL': '🚨'
        }
        
        # Лимиты отправки
        self.rate_limit = {
            'trades': 10,  # Максимум 10 уведомлений о сделках в минуту
            'portfolio': 1,  # Максимум 1 обновление портфеля в минуту
            'alerts': 5   # Максимум 5 алертов в минуту
        }
        
        self.last_sent = {
            'trades': [],
            'portfolio': None,
            'alerts': []
        }
    
    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, notification_type: str) -> bool:
        """Проверка лимитов отправки"""
        now = datetime.now()
        
        if notification_type == 'trades':
            # Удаляем старые записи (старше 1 минуты)
            self.last_sent['trades'] = [
                timestamp for timestamp in self.last_sent['trades']
                if (now - timestamp).seconds < 60
            ]
            return len(self.last_sent['trades']) < self.rate_limit['trades']
        
        elif notification_type == 'portfolio':
            if self.last_sent['portfolio'] is None:
                return True
            return (now - self.last_sent['portfolio']).seconds >= 60
        
        elif notification_type == 'alerts':
            # Удаляем старые записи (старше 1 минуты)
            self.last_sent['alerts'] = [
                timestamp for timestamp in self.last_sent['alerts']
                if (now - timestamp).seconds < 60
            ]
            return len(self.last_sent['alerts']) < self.rate_limit['alerts']
        
        return True
    
    def _update_rate_limit(self, notification_type: str):
        """Обновление счетчика лимитов"""
        now = datetime.now()
        
        if notification_type == 'trades':
            self.last_sent['trades'].append(now)
        elif notification_type == 'portfolio':
            self.last_sent['portfolio'] = now
        elif notification_type == 'alerts':
            self.last_sent['alerts'].append(now)
    
    def _format_trade_message(self, trade: TradeNotification) -> str:
        """Форматирование сообщения о сделке"""
        emoji = self.emoji_map.get(trade.action, '📊')
        reason_emoji = self.emoji_map.get(trade.reason, '') if trade.reason else ''
        
        message = f"{emoji} **{trade.action}** {trade.symbol}\n"
        message += f"📊 Стратегия: {trade.strategy}\n"
        message += f"💰 Цена: {trade.price:.2f}₽\n"
        message += f"📦 Количество: {trade.quantity} шт.\n"
        message += f"💵 Сумма: {trade.amount:,.0f}₽\n"
        
        if trade.pnl is not None:
            pnl_emoji = "📈" if trade.pnl >= 0 else "📉"
            message += f"{pnl_emoji} P&L: {trade.pnl:+,.0f}₽\n"
        
        if trade.reason:
            message += f"{reason_emoji} Причина: {trade.reason}\n"
        
        if trade.confidence is not None:
            confidence_emoji = "🎯" if trade.confidence > 0.7 else "⚠️"
            message += f"{confidence_emoji} Уверенность: {trade.confidence:.1%}\n"
        
        message += f"🕐 Время: {trade.timestamp}"
        
        return message
    
    def _format_portfolio_message(self, portfolio: PortfolioUpdate) -> str:
        """Форматирование сообщения о портфеле"""
        pnl_emoji = "📈" if portfolio.total_pnl >= 0 else "📉"
        daily_emoji = "📈" if portfolio.daily_pnl >= 0 else "📉"
        
        message = f"📊 **ОБНОВЛЕНИЕ ПОРТФЕЛЯ**\n\n"
        message += f"💰 Общая стоимость: {portfolio.total_value:,.0f}₽\n"
        message += f"{pnl_emoji} Общий P&L: {portfolio.total_pnl:+,.0f}₽ ({portfolio.total_pnl_pct:+.2f}%)\n"
        message += f"{daily_emoji} Дневной P&L: {portfolio.daily_pnl:+,.0f}₽ ({portfolio.daily_pnl_pct:+.2f}%)\n"
        message += f"📦 Позиций: {portfolio.positions_count}\n"
        message += f"🕐 Время: {portfolio.timestamp}"
        
        return message
    
    def _format_alert_message(self, alert: AlertNotification) -> str:
        """Форматирование алерт-сообщения"""
        emoji = self.emoji_map.get(alert.alert_type, 'ℹ️')
        severity_emoji = self.emoji_map.get(alert.severity, '')
        
        message = f"{emoji} **{alert.title}**\n"
        message += f"{severity_emoji} Тип: {alert.alert_type}\n"
        message += f"📝 {alert.message}\n"
        message += f"🕐 Время: {alert.timestamp}"
        
        return message
    
    async def _send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Отправка сообщения в Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot token или chat_id не настроены")
            return False
        
        if not self.session:
            logger.error("HTTP сессия не инициализирована")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    logger.info("Сообщение успешно отправлено в Telegram")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка отправки в Telegram: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения в Telegram: {e}")
            return False
    
    async def send_trade_notification(self, trade: TradeNotification) -> bool:
        """Отправка уведомления о сделке"""
        if not self._check_rate_limit('trades'):
            logger.warning("Превышен лимит отправки уведомлений о сделках")
            return False
        
        message = self._format_trade_message(trade)
        success = await self._send_message(message)
        
        if success:
            self._update_rate_limit('trades')
        
        return success
    
    async def send_portfolio_update(self, portfolio: PortfolioUpdate) -> bool:
        """Отправка обновления портфеля"""
        if not self._check_rate_limit('portfolio'):
            logger.warning("Превышен лимит отправки обновлений портфеля")
            return False
        
        message = self._format_portfolio_message(portfolio)
        success = await self._send_message(message)
        
        if success:
            self._update_rate_limit('portfolio')
        
        return success
    
    async def send_alert(self, alert: AlertNotification) -> bool:
        """Отправка алерта"""
        if not self._check_rate_limit('alerts'):
            logger.warning("Превышен лимит отправки алертов")
            return False
        
        message = self._format_alert_message(alert)
        success = await self._send_message(message)
        
        if success:
            self._update_rate_limit('alerts')
        
        return success
    
    async def send_daily_summary(self, trades: List[TradeNotification], 
                                portfolio: PortfolioUpdate) -> bool:
        """Отправка дневной сводки"""
        if not trades:
            return False
        
        # Подсчет статистики
        total_trades = len(trades)
        buy_trades = len([t for t in trades if t.action == 'BUY'])
        sell_trades = len([t for t in trades if t.action == 'SELL'])
        total_pnl = sum(t.pnl or 0 for t in trades)
        profitable_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        
        message = f"📊 **ДНЕВНАЯ СВОДКА**\n\n"
        message += f"📈 Всего сделок: {total_trades}\n"
        message += f"🟢 Покупок: {buy_trades}\n"
        message += f"🔴 Продаж: {sell_trades}\n"
        message += f"💰 Общий P&L: {total_pnl:+,.0f}₽\n"
        message += f"✅ Прибыльных сделок: {profitable_trades}/{total_trades}\n"
        message += f"📊 Стоимость портфеля: {portfolio.total_value:,.0f}₽\n"
        message += f"🕐 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self._send_message(message)

class TradingNotifier:
    """Основной класс для уведомлений о торговле"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.notifier = TelegramNotifier(bot_token, chat_id)
        self.trades_history = []
        self.portfolio_history = []
    
    async def notify_trade(self, symbol: str, action: str, quantity: int, 
                          price: float, strategy: str, pnl: float = None,
                          reason: str = None, confidence: float = None):
        """Уведомление о сделке"""
        trade = TradeNotification(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            amount=quantity * price,
            strategy=strategy,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            pnl=pnl,
            reason=reason,
            confidence=confidence
        )
        
        self.trades_history.append(trade)
        
        async with self.notifier as notifier:
            await notifier.send_trade_notification(trade)
    
    async def notify_portfolio_update(self, total_value: float, total_pnl: float,
                                    positions_count: int, daily_pnl: float = 0.0):
        """Уведомление об обновлении портфеля"""
        portfolio = PortfolioUpdate(
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0,
            positions_count=positions_count,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            daily_pnl=daily_pnl,
            daily_pnl_pct=(daily_pnl / total_value) * 100 if total_value > 0 else 0
        )
        
        self.portfolio_history.append(portfolio)
        
        async with self.notifier as notifier:
            await notifier.send_portfolio_update(portfolio)
    
    async def notify_alert(self, alert_type: str, title: str, message: str,
                          severity: str = 'INFO'):
        """Уведомление об алерте"""
        alert = AlertNotification(
            alert_type=alert_type,
            title=title,
            message=message,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            severity=severity
        )
        
        async with self.notifier as notifier:
            await notifier.send_alert(alert)
    
    async def send_daily_summary(self):
        """Отправка дневной сводки"""
        if not self.trades_history or not self.portfolio_history:
            return False
        
        latest_portfolio = self.portfolio_history[-1]
        
        async with self.notifier as notifier:
            return await notifier.send_daily_summary(self.trades_history, latest_portfolio)

# Пример использования
async def main():
    """Пример использования системы уведомлений"""
    
    # Создаем уведомлятель
    notifier = TradingNotifier()
    
    # Симулируем сделки
    await notifier.notify_trade(
        symbol="GAZP",
        action="BUY",
        quantity=100,
        price=150.50,
        strategy="ARIMA",
        confidence=0.85
    )
    
    await notifier.notify_trade(
        symbol="GAZP",
        action="SELL",
        quantity=100,
        price=152.30,
        strategy="ARIMA",
        pnl=180.0,
        reason="TAKE_PROFIT"
    )
    
    # Обновление портфеля
    await notifier.notify_portfolio_update(
        total_value=105000,
        total_pnl=5000,
        positions_count=3,
        daily_pnl=180.0
    )
    
    # Алерт
    await notifier.notify_alert(
        alert_type="RISK",
        title="Превышен лимит риска",
        message="Текущий риск портфеля превышает 15%",
        severity="HIGH"
    )
    
    # Дневная сводка
    await notifier.send_daily_summary()

if __name__ == "__main__":
    asyncio.run(main())
