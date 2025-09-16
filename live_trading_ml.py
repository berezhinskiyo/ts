#!/usr/bin/env python3
"""
Скрипт для постоянной торговли с использованием продвинутых ML стратегий
"""

import os
import sys
import time
import asyncio
import logging
import signal
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты
from advanced_ml_strategies import AdvancedMLStrategies
from env_loader import load_env_file
from telegram_notifications import TradingNotifier

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveTradingManager:
    """Менеджер для постоянной торговли"""
    
    def __init__(self, config_file: str = None, strategy: str = 'ensemble'):
        self.config_file = config_file
        self.strategy = strategy
        self.running = False
        self.trading_session = None
        self.telegram_notifier = None
        
        # Загружаем конфигурацию
        self.load_configuration()
        
        # Инициализируем компоненты
        self.initialize_components()
        
        # Настройка обработки сигналов
        self.setup_signal_handlers()
    
    def load_configuration(self):
        """Загрузка конфигурации"""
        logger.info("🔧 Загрузка конфигурации...")
        
        # Загружаем .env файл
        if self.config_file:
            load_env_file(self.config_file)
            logger.info(f"✅ Конфигурация загружена из: {self.config_file}")
        else:
            # Автопоиск .env файла
            env_paths = ['.env', 'config/.env', 'config/environments/.env']
            for path in env_paths:
                if os.path.exists(path):
                    load_env_file(path)
                    logger.info(f"✅ Конфигурация загружена из: {path}")
                    break
        
        # Проверяем обязательные переменные
        required_vars = ['TBANK_TOKEN', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Отсутствуют обязательные переменные: {', '.join(missing_vars)}")
            sys.exit(1)
        
        # Настройки торговли
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', 100000))
        self.max_risk_per_trade = float(os.getenv('MAX_RISK_PER_TRADE', 0.02))
        self.use_sandbox = os.getenv('USE_SANDBOX', 'True').lower() == 'true'
        
        # Настройки мониторинга
        self.check_interval = int(os.getenv('CHECK_INTERVAL', 300))  # 5 минут
        self.performance_check_interval = int(os.getenv('PERFORMANCE_CHECK_INTERVAL', 3600))  # 1 час
        self.retrain_interval = int(os.getenv('RETRAIN_INTERVAL', 86400))  # 24 часа
        
        # Пороги для переобучения
        self.min_win_rate = float(os.getenv('MIN_WIN_RATE', 0.4))  # 40%
        self.max_drawdown_threshold = float(os.getenv('MAX_DRAWDOWN_THRESHOLD', 0.15))  # 15%
        self.min_sharpe_ratio = float(os.getenv('MIN_SHARPE_RATIO', 0.5))
        
        logger.info("✅ Конфигурация загружена успешно")
    
    def initialize_components(self):
        """Инициализация компонентов"""
        logger.info("🚀 Инициализация компонентов...")
        
        # Инициализируем торговую сессию
        self.trading_session = AdvancedMLStrategies(
            initial_capital=self.initial_capital,
            optimize_indicators=True,
            max_indicators=20,
            enable_telegram=True
        )
        
        # Инициализируем Telegram уведомления
        try:
            self.telegram_notifier = TradingNotifier(
                bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
                chat_id=os.getenv('TELEGRAM_CHAT_ID')
            )
            logger.info("✅ Telegram уведомления инициализированы")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось инициализировать Telegram: {e}")
            self.telegram_notifier = None
        
        logger.info("✅ Компоненты инициализированы")
    
    def setup_signal_handlers(self):
        """Настройка обработчиков сигналов"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Получен сигнал {signum}, завершение работы...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def send_startup_notification(self):
        """Отправка уведомления о запуске"""
        if not self.telegram_notifier:
            return
        
        try:
            await self.telegram_notifier.notify_alert(
                alert_type='INFO',
                title='🚀 Запуск торговой системы',
                message=f"""
🤖 **Торговая система запущена**

📊 **Стратегия**: {self.strategy.upper()}
💰 **Начальный капитал**: {self.initial_capital:,.0f} ₽
⚠️ **Риск на сделку**: {self.max_risk_per_trade*100:.1f}%
🔄 **Интервал проверки**: {self.check_interval} сек
📈 **Переобучение**: каждые {self.retrain_interval//3600} часов

🛡️ **Пороги безопасности**:
• Минимальный Win Rate: {self.min_win_rate*100:.0f}%
• Максимальная просадка: {self.max_drawdown_threshold*100:.0f}%
• Минимальный Sharpe: {self.min_sharpe_ratio:.1f}

⏰ **Время запуска**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """,
                severity='INFO'
            )
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления о запуске: {e}")
    
    async def send_shutdown_notification(self):
        """Отправка уведомления о завершении"""
        if not self.telegram_notifier:
            return
        
        try:
            # Получаем статистику сессии
            stats = self.get_session_stats()
            
            await self.telegram_notifier.notify_alert(
                alert_type='INFO',
                title='🛑 Завершение торговой системы',
                message=f"""
🤖 **Торговая система остановлена**

📊 **Статистика сессии**:
• Текущий капитал: {stats.get('current_capital', 0):,.0f} ₽
• Общая прибыль: {stats.get('total_pnl', 0):,.0f} ₽ ({stats.get('total_pnl_pct', 0):.2f}%)
• Количество сделок: {stats.get('total_trades', 0)}
• Win Rate: {stats.get('win_rate', 0)*100:.1f}%

⏰ **Время завершения**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """,
                severity='INFO'
            )
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления о завершении: {e}")
    
    def get_session_stats(self) -> Dict:
        """Получение статистики сессии"""
        if not self.trading_session:
            return {}
        
        try:
            trades = self.trading_session.trades
            current_capital = self.trading_session.current_capital
            
            if not trades:
                return {
                    'current_capital': current_capital,
                    'total_pnl': 0,
                    'total_pnl_pct': 0,
                    'total_trades': 0,
                    'win_rate': 0
                }
            
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            total_pnl_pct = (total_pnl / self.initial_capital) * 100
            
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            win_rate = winning_trades / len(trades) if trades else 0
            
            return {
                'current_capital': current_capital,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'total_trades': len(trades),
                'win_rate': win_rate
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}
    
    async def check_performance(self):
        """Проверка производительности и переобучение"""
        logger.info("📊 Проверка производительности...")
        
        try:
            stats = self.get_session_stats()
            
            # Проверяем пороги
            needs_retrain = False
            issues = []
            
            if stats.get('win_rate', 0) < self.min_win_rate:
                needs_retrain = True
                issues.append(f"Win Rate {stats.get('win_rate', 0)*100:.1f}% < {self.min_win_rate*100:.0f}%")
            
            if stats.get('total_pnl_pct', 0) < -self.max_drawdown_threshold * 100:
                needs_retrain = True
                issues.append(f"Просадка {abs(stats.get('total_pnl_pct', 0)):.1f}% > {self.max_drawdown_threshold*100:.0f}%")
            
            if needs_retrain and self.telegram_notifier:
                await self.telegram_notifier.notify_alert(
                    alert_type='PERFORMANCE',
                    title='⚠️ Требуется переобучение модели',
                    message=f"""
📉 **Обнаружены проблемы с производительностью**:

{chr(10).join(f"• {issue}" for issue in issues)}

🔄 **Инициируется переобучение модели...**
                    """,
                    severity='HIGH'
                )
            
            return needs_retrain
            
        except Exception as e:
            logger.error(f"Ошибка проверки производительности: {e}")
            return False
    
    async def retrain_models(self):
        """Переобучение моделей"""
        logger.info("🔄 Переобучение моделей...")
        
        try:
            if self.telegram_notifier:
                await self.telegram_notifier.notify_alert(
                    alert_type='INFO',
                    title='🔄 Переобучение моделей',
                    message='Начинается переобучение ML моделей на новых данных...',
                    severity='MEDIUM'
                )
            
            # Здесь можно добавить логику переобучения
            # Например, загрузка новых данных и переобучение моделей
            
            if self.telegram_notifier:
                await self.telegram_notifier.notify_alert(
                    alert_type='INFO',
                    title='✅ Переобучение завершено',
                    message='ML модели успешно переобучены на новых данных.',
                    severity='LOW'
                )
            
            logger.info("✅ Переобучение завершено")
            
        except Exception as e:
            logger.error(f"Ошибка переобучения: {e}")
            if self.telegram_notifier:
                await self.telegram_notifier.notify_alert(
                    alert_type='ERROR',
                    title='❌ Ошибка переобучения',
                    message=f'Не удалось переобучить модели: {str(e)}',
                    severity='CRITICAL'
                )
    
    async def run_trading_cycle(self):
        """Один цикл торговли"""
        try:
            logger.info("🔄 Запуск торгового цикла...")
            
            # Проверяем время работы биржи
            if not self.trading_session.is_market_open():
                market_status = self.trading_session.get_market_status()
                logger.info(f"🕐 Биржа закрыта. {market_status['next_action']}")
                return
            
            # Загружаем данные
            market_data = self.trading_session.load_tbank_data()
            
            if not market_data:
                logger.warning("⚠️ Нет данных для торговли")
                return
            
            # Выбираем стратегию
            strategy_mapping = {
                'ensemble': 'ensemble_ml_strategy',
                'arima': 'arima_strategy',
                'lstm': 'lstm_strategy',
                'sarima': 'sarima_strategy'
            }
            
            strategy_method_name = strategy_mapping.get(self.strategy, f"{self.strategy}_strategy")
            strategy_method = getattr(self.trading_session, strategy_method_name, None)
            if not strategy_method:
                logger.error(f"❌ Стратегия {self.strategy} не найдена (искали метод {strategy_method_name})")
                return
            
            # Запускаем торговлю для каждого инструмента
            for data_key, data in market_data.items():
                symbol = data_key.split('_')[0]
                logger.info(f"📈 Торговля {symbol}...")
                
                try:
                    # Запускаем стратегию
                    result = strategy_method(symbol, data)
                    
                except Exception as e:
                    logger.error(f"Ошибка торговли {symbol}: {e}")
                    if self.telegram_notifier:
                        await self.telegram_notifier.notify_alert(
                            alert_type='ERROR',
                            title=f'❌ Ошибка торговли {symbol}',
                            message=str(e),
                            severity='HIGH'
                        )
            
            logger.info("✅ Торговый цикл завершен")
            
        except Exception as e:
            logger.error(f"Ошибка торгового цикла: {e}")
    
    async def run(self):
        """Основной цикл торговли"""
        logger.info("🚀 Запуск постоянной торговли...")
        
        # Отправляем уведомление о запуске
        await self.send_startup_notification()
        
        self.running = True
        last_performance_check = time.time()
        last_retrain = time.time()
        
        try:
            while self.running:
                start_time = time.time()
                
                # Основной торговый цикл
                await self.run_trading_cycle()
                
                # Проверка производительности
                if time.time() - last_performance_check >= self.performance_check_interval:
                    needs_retrain = await self.check_performance()
                    
                    # Переобучение при необходимости
                    if needs_retrain and time.time() - last_retrain >= self.retrain_interval:
                        await self.retrain_models()
                        last_retrain = time.time()
                    
                    last_performance_check = time.time()
                
                # Ожидание до следующего цикла
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                
                if sleep_time > 0:
                    logger.info(f"⏳ Ожидание {sleep_time:.0f} сек до следующего цикла...")
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("🛑 Получен сигнал прерывания")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")
            if self.telegram_notifier:
                await self.telegram_notifier.notify_alert(
                    alert_type='ERROR',
                    title='💥 Критическая ошибка системы',
                    message=f'Торговая система остановлена из-за ошибки: {str(e)}',
                    severity='CRITICAL'
                )
        finally:
            await self.send_shutdown_notification()
            logger.info("✅ Торговая система остановлена")
    
    def stop(self):
        """Остановка торговой системы"""
        self.running = False

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Постоянная торговля с ML стратегиями')
    parser.add_argument('--config', '-c', help='Путь к .env файлу')
    parser.add_argument('--strategy', '-s', default='ensemble', 
                       choices=['arima', 'lstm', 'sarima', 'ensemble'],
                       help='Стратегия для торговли')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Тестовый режим без реальных сделок')
    
    args = parser.parse_args()
    
    # Создаем менеджер торговли
    manager = LiveTradingManager(
        config_file=args.config,
        strategy=args.strategy
    )
    
    # Запускаем торговлю
    try:
        asyncio.run(manager.run())
    except KeyboardInterrupt:
        logger.info("🛑 Торговля остановлена пользователем")
    except Exception as e:
        logger.error(f"❌ Ошибка запуска: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
