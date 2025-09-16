#!/usr/bin/env python3
"""
Скрипт для проверки готовности системы к реальной торговле
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env_file
from telegram_notifications import TradingNotifier

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingReadinessChecker:
    """Проверка готовности системы к торговле"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.checks_passed = 0
        self.checks_failed = 0
        self.issues = []
        self.warnings = []
        
    def load_configuration(self):
        """Загрузка конфигурации"""
        logger.info("🔧 Загрузка конфигурации...")
        
        try:
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
                else:
                    self.add_issue("❌ Не найден .env файл")
                    return False
            
            return True
        except Exception as e:
            self.add_issue(f"❌ Ошибка загрузки конфигурации: {e}")
            return False
    
    def add_issue(self, issue: str):
        """Добавление критической проблемы"""
        self.issues.append(issue)
        self.checks_failed += 1
        logger.error(issue)
    
    def add_warning(self, warning: str):
        """Добавление предупреждения"""
        self.warnings.append(warning)
        logger.warning(warning)
    
    def add_success(self, message: str):
        """Добавление успешной проверки"""
        self.checks_passed += 1
        logger.info(f"✅ {message}")
    
    def check_environment_variables(self):
        """Проверка переменных окружения"""
        logger.info("\n📋 Проверка переменных окружения...")
        
        # Обязательные переменные
        required_vars = {
            'TBANK_TOKEN': 'Токен T-Bank API',
            'TELEGRAM_BOT_TOKEN': 'Токен Telegram бота',
            'TELEGRAM_CHAT_ID': 'Chat ID для уведомлений'
        }
        
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                self.add_issue(f"❌ Отсутствует {var} ({description})")
            else:
                self.add_success(f"{var} настроен")
        
        # Проверка настроек торговли
        trading_vars = {
            'INITIAL_CAPITAL': (100000, 'Начальный капитал'),
            'MAX_RISK_PER_TRADE': (0.02, 'Максимальный риск на сделку'),
            'STOP_LOSS_PERCENTAGE': (0.05, 'Процент стоп-лосса'),
            'TAKE_PROFIT_PERCENTAGE': (0.15, 'Процент тейк-профита'),
            'TRADING_SYMBOLS': ('GAZP,SBER,PIKK,IRAO,SGZH', 'Список инструментов для торговли'),
            'TRADING_PERIOD': ('1Y', 'Период данных для анализа'),
            'MIN_DATA_DAYS': (100, 'Минимальное количество дней данных')
        }
        
        for var, (default, description) in trading_vars.items():
            value = os.getenv(var)
            if value:
                # Для числовых переменных проверяем, что это число
                if var in ['INITIAL_CAPITAL', 'MAX_RISK_PER_TRADE', 'STOP_LOSS_PERCENTAGE', 'TAKE_PROFIT_PERCENTAGE', 'MIN_DATA_DAYS']:
                    try:
                        float_value = float(value)
                        self.add_success(f"{var}: {float_value} ({description})")
                    except ValueError:
                        self.add_issue(f"❌ Неверное значение {var}: {value}")
                else:
                    # Для строковых переменных просто показываем значение
                    self.add_success(f"{var}: {value} ({description})")
            else:
                self.add_warning(f"⚠️ {var} не установлен, используется значение по умолчанию: {default}")
    
    def check_trading_mode(self):
        """Проверка режима торговли"""
        logger.info("\n🚨 Проверка режима торговли...")
        
        use_sandbox = os.getenv('USE_SANDBOX', 'True').lower()
        
        if use_sandbox == 'false':
            self.add_success("Режим реальной торговли включен (USE_SANDBOX=False)")
            
            # Дополнительные проверки для реальной торговли
            initial_capital = float(os.getenv('INITIAL_CAPITAL', 100000))
            if initial_capital > 1000000:
                self.add_warning(f"⚠️ Большой начальный капитал: {initial_capital:,.0f} ₽")
            
            max_risk = float(os.getenv('MAX_RISK_PER_TRADE', 0.02))
            if max_risk > 0.05:
                self.add_warning(f"⚠️ Высокий риск на сделку: {max_risk*100:.1f}%")
                
        else:
            self.add_issue("❌ Включен режим песочницы (USE_SANDBOX=True). Для реальной торговли установите USE_SANDBOX=False")
    
    def check_risk_management(self):
        """Проверка управления рисками"""
        logger.info("\n🛡️ Проверка управления рисками...")
        
        # Проверка стоп-лосса
        stop_loss = float(os.getenv('STOP_LOSS_PERCENTAGE', 0.05))
        if stop_loss <= 0 or stop_loss > 0.2:
            self.add_issue(f"❌ Некорректный стоп-лосс: {stop_loss*100:.1f}% (должен быть 0-20%)")
        else:
            self.add_success(f"Стоп-лосс: {stop_loss*100:.1f}%")
        
        # Проверка тейк-профита
        take_profit = float(os.getenv('TAKE_PROFIT_PERCENTAGE', 0.15))
        if take_profit <= 0 or take_profit > 1.0:
            self.add_issue(f"❌ Некорректный тейк-профит: {take_profit*100:.1f}% (должен быть 0-100%)")
        else:
            self.add_success(f"Тейк-профит: {take_profit*100:.1f}%")
        
        # Проверка соотношения риск/прибыль
        if take_profit / stop_loss < 1.5:
            self.add_warning(f"⚠️ Низкое соотношение риск/прибыль: {take_profit/stop_loss:.1f}")
        else:
            self.add_success(f"Соотношение риск/прибыль: {take_profit/stop_loss:.1f}")
        
        # Проверка максимального риска на сделку
        max_risk = float(os.getenv('MAX_RISK_PER_TRADE', 0.02))
        if max_risk > 0.05:
            self.add_issue(f"❌ Слишком высокий риск на сделку: {max_risk*100:.1f}% (рекомендуется ≤5%)")
        else:
            self.add_success(f"Максимальный риск на сделку: {max_risk*100:.1f}%")
    
    async def check_telegram_notifications(self):
        """Проверка Telegram уведомлений"""
        logger.info("\n📱 Проверка Telegram уведомлений...")
        
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                self.add_issue("❌ Не настроены Telegram токены")
                return
            
            # Создаем уведомлятель
            notifier = TradingNotifier(bot_token=bot_token, chat_id=chat_id)
            
            # Тестируем отправку уведомления
            await notifier.notify_alert(
                alert_type='INFO',
                title='🔍 Проверка системы',
                message=f'Тестовое уведомление от системы проверки готовности\nВремя: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                severity='LOW'
            )
            
            self.add_success("Telegram уведомления работают")
            
        except Exception as e:
            self.add_issue(f"❌ Ошибка Telegram уведомлений: {e}")
    
    def check_dependencies(self):
        """Проверка зависимостей"""
        logger.info("\n📦 Проверка зависимостей...")
        
        required_packages = [
            'pandas', 'numpy', 'sklearn', 'statsmodels', 
            'aiohttp', 'dotenv'
        ]
        
        optional_packages = [
            'tensorflow', 'keras'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.add_success(f"Пакет {package} установлен")
            except ImportError:
                self.add_issue(f"❌ Отсутствует обязательный пакет: {package}")
        
        for package in optional_packages:
            try:
                __import__(package)
                self.add_success(f"Пакет {package} установлен (опциональный)")
            except ImportError:
                self.add_warning(f"⚠️ Отсутствует опциональный пакет: {package} (LSTM стратегии недоступны)")
    
    def check_file_permissions(self):
        """Проверка прав доступа к файлам"""
        logger.info("\n📁 Проверка прав доступа...")
        
        files_to_check = [
            'live_trading_ml.py',
            'advanced_ml_strategies.py',
            'telegram_notifications.py',
            'env_loader.py'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                if os.access(file_path, os.R_OK):
                    self.add_success(f"Файл {file_path} доступен для чтения")
                else:
                    self.add_issue(f"❌ Нет прав на чтение файла: {file_path}")
            else:
                self.add_issue(f"❌ Файл не найден: {file_path}")
    
    def check_trading_instruments(self):
        """Проверка доступности инструментов для торговли"""
        logger.info("\n📈 Проверка инструментов для торговли...")
        
        try:
            # Получаем настройки
            trading_symbols = os.getenv('TRADING_SYMBOLS', 'GAZP,SBER,PIKK,IRAO,SGZH').split(',')
            trading_period = os.getenv('TRADING_PERIOD', '1Y')
            min_data_days = int(os.getenv('MIN_DATA_DAYS', '100'))
            
            self.add_success(f"Настроенные инструменты: {', '.join(trading_symbols)}")
            self.add_success(f"Период данных: {trading_period}")
            self.add_success(f"Минимум дней данных: {min_data_days}")
            
            # Проверяем доступность данных
            data_dir = 'data/tbank_real'
            if not os.path.exists(data_dir):
                self.add_issue(f"❌ Директория с данными не найдена: {data_dir}")
                return
            
            available_instruments = []
            for symbol in trading_symbols:
                filename = f"{symbol}_{trading_period}_tbank.csv"
                filepath = os.path.join(data_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        import pandas as pd
                        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
                        if len(df) >= min_data_days:
                            available_instruments.append(symbol)
                            self.add_success(f"✅ {symbol}: {len(df)} дней данных")
                        else:
                            self.add_warning(f"⚠️ {symbol}: недостаточно данных ({len(df)} < {min_data_days})")
                    except Exception as e:
                        self.add_issue(f"❌ Ошибка чтения данных {symbol}: {e}")
                else:
                    self.add_issue(f"❌ Файл данных не найден: {filename}")
            
            if available_instruments:
                self.add_success(f"Доступно для торговли: {', '.join(available_instruments)}")
            else:
                self.add_issue("❌ Нет доступных инструментов для торговли")
                
        except Exception as e:
            self.add_issue(f"❌ Ошибка проверки инструментов: {e}")
    
    def check_logging_setup(self):
        """Проверка настройки логирования"""
        logger.info("\n📝 Проверка логирования...")
        
        log_files = ['live_trading.log', 'trading.log']
        
        for log_file in log_files:
            try:
                # Проверяем, можем ли мы создать/записать в лог файл
                with open(log_file, 'a') as f:
                    f.write(f"# Test log entry - {datetime.now()}\n")
                self.add_success(f"Лог файл {log_file} доступен для записи")
            except Exception as e:
                self.add_warning(f"⚠️ Проблема с лог файлом {log_file}: {e}")
    
    def generate_recommendations(self):
        """Генерация рекомендаций"""
        logger.info("\n💡 Рекомендации...")
        
        recommendations = []
        
        # Рекомендации по безопасности
        if self.checks_failed == 0:
            recommendations.append("✅ Система готова к торговле!")
        else:
            recommendations.append("❌ Исправьте критические проблемы перед запуском")
        
        # Рекомендации по настройкам
        initial_capital = float(os.getenv('INITIAL_CAPITAL', 100000))
        if initial_capital > 500000:
            recommendations.append("💡 Рекомендуется начать с меньшего капитала для тестирования")
        
        max_risk = float(os.getenv('MAX_RISK_PER_TRADE', 0.02))
        if max_risk > 0.03:
            recommendations.append("💡 Рекомендуется снизить риск на сделку до 2-3%")
        
        # Рекомендации по мониторингу
        recommendations.append("💡 Настройте регулярный мониторинг логов и уведомлений")
        recommendations.append("💡 Протестируйте систему в тестовом режиме перед реальной торговлей")
        
        for rec in recommendations:
            logger.info(rec)
        
        return recommendations
    
    async def run_all_checks(self):
        """Запуск всех проверок"""
        logger.info("🔍 ПРОВЕРКА ГОТОВНОСТИ СИСТЕМЫ К ТОРГОВЛЕ")
        logger.info("=" * 60)
        
        # Загружаем конфигурацию
        if not self.load_configuration():
            return False
        
        # Выполняем все проверки
        self.check_environment_variables()
        self.check_trading_mode()
        self.check_risk_management()
        await self.check_telegram_notifications()
        self.check_dependencies()
        self.check_file_permissions()
        self.check_trading_instruments()
        self.check_logging_setup()
        
        # Генерируем рекомендации
        recommendations = self.generate_recommendations()
        
        # Выводим итоговый отчет
        self.print_summary()
        
        return self.checks_failed == 0
    
    def print_summary(self):
        """Вывод итогового отчета"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 ИТОГОВЫЙ ОТЧЕТ")
        logger.info("=" * 60)
        
        logger.info(f"✅ Проверок пройдено: {self.checks_passed}")
        logger.info(f"❌ Проверок провалено: {self.checks_failed}")
        logger.info(f"⚠️ Предупреждений: {len(self.warnings)}")
        
        if self.issues:
            logger.info("\n🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
            for issue in self.issues:
                logger.info(f"  {issue}")
        
        if self.warnings:
            logger.info("\n⚠️ ПРЕДУПРЕЖДЕНИЯ:")
            for warning in self.warnings:
                logger.info(f"  {warning}")
        
        if self.checks_failed == 0:
            logger.info("\n🎉 СИСТЕМА ГОТОВА К ТОРГОВЛЕ!")
            logger.info("🚀 Запустите: python live_trading_ml.py --config config/environments/.env --strategy ensemble")
        else:
            logger.info("\n❌ СИСТЕМА НЕ ГОТОВА К ТОРГОВЛЕ")
            logger.info("🔧 Исправьте критические проблемы и повторите проверку")

async def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Проверка готовности системы к торговле')
    parser.add_argument('--config', '-c', help='Путь к .env файлу')
    
    args = parser.parse_args()
    
    checker = TradingReadinessChecker(config_file=args.config)
    success = await checker.run_all_checks()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
