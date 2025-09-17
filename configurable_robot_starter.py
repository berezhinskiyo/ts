#!/usr/bin/env python3
"""
Конфигурируемый стартер для торговых роботов
Поддерживает все реализованные стратегии с анализом новостей
"""

import os
import sys
import json
import time
import asyncio
import logging
import signal
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты стратегий
from advanced_ml_strategies import AdvancedMLStrategies
from russian_news_analyzer import RussianNewsAnalyzer
from russian_trading_integration import RussianTradingStrategy
from news_data_manager import NewsDataManager
from telegram_notifications import TradingNotifier
from env_loader import load_env_file

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robot_starter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategyRegistry:
    """Реестр всех доступных стратегий"""
    
    def __init__(self):
        self.strategies = {
            # ML стратегии
            'random_forest': {
                'name': 'Random Forest',
                'type': 'ml',
                'description': 'Стратегия на основе Random Forest',
                'performance_without_news': 0.0,  # Будет обновлено из результатов
                'performance_with_news': 2.30,    # Будет обновлено из результатов
                'improvement': 2.30,
                'class': 'RandomForestStrategy'
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'type': 'ml',
                'description': 'Стратегия на основе Gradient Boosting',
                'performance_without_news': 0.0,
                'performance_with_news': 0.0,
                'improvement': 0.0,
                'class': 'GradientBoostingStrategy'
            },
            'ridge_regression': {
                'name': 'Ridge Regression',
                'type': 'ml',
                'description': 'Стратегия на основе Ridge регрессии',
                'performance_without_news': 0.0,
                'performance_with_news': 0.0,
                'improvement': 0.0,
                'class': 'RidgeStrategy'
            },
            'linear_regression': {
                'name': 'Linear Regression',
                'type': 'ml',
                'description': 'Стратегия на основе линейной регрессии',
                'performance_without_news': 0.0,
                'performance_with_news': 0.0,
                'improvement': 0.0,
                'class': 'LinearRegressionStrategy'
            },
            'ensemble': {
                'name': 'Ensemble',
                'type': 'ml',
                'description': 'Ансамблевая стратегия',
                'performance_without_news': 0.0,
                'performance_with_news': 13.00,   # Среднее улучшение
                'improvement': 13.00,
                'class': 'EnsembleStrategy'
            },
            # Технические стратегии
            'technical': {
                'name': 'Technical Analysis',
                'type': 'technical',
                'description': 'Техническая стратегия на основе индикаторов',
                'performance_without_news': 0.0,
                'performance_with_news': -5.43,   # Среднее ухудшение
                'improvement': -5.43,
                'class': 'TechnicalStrategy'
            },
            'momentum': {
                'name': 'Momentum Strategy',
                'type': 'technical',
                'description': 'Стратегия следования за трендом',
                'performance_without_news': 0.0,
                'performance_with_news': 0.0,
                'improvement': 0.0,
                'class': 'MomentumStrategy'
            },
            'mean_reversion': {
                'name': 'Mean Reversion',
                'type': 'technical',
                'description': 'Стратегия возврата к среднему',
                'performance_without_news': 0.0,
                'performance_with_news': 0.0,
                'improvement': 0.0,
                'class': 'MeanReversionStrategy'
            },
            # Комбинированные стратегии
            'ml_with_news': {
                'name': 'ML with News Analysis',
                'type': 'combined',
                'description': 'ML стратегия с анализом новостей',
                'performance_without_news': 0.0,
                'performance_with_news': 13.00,
                'improvement': 13.00,
                'class': 'MLWithNewsStrategy'
            },
            'technical_with_news': {
                'name': 'Technical with News Analysis',
                'type': 'combined',
                'description': 'Техническая стратегия с анализом новостей',
                'performance_without_news': 0.0,
                'performance_with_news': -5.43,
                'improvement': -5.43,
                'class': 'TechnicalWithNewsStrategy'
            }
        }
        
        # Загружаем актуальные показатели из результатов тестирования
        self.load_performance_data()
    
    def load_performance_data(self):
        """Загрузка актуальных показателей производительности"""
        try:
            # Загружаем результаты 3-летнего тестирования
            results_file = 'quick_3year_backtesting_results.json'
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # Обновляем показатели для ML стратегии
                if 'ML' in results.get('results', {}):
                    ml_results = results['results']['ML']
                    total_improvement = 0.0
                    count = 0
                    
                    for symbol, data in ml_results.items():
                        improvement = data.get('improvements', {}).get('return_improvement', 0.0)
                        total_improvement += improvement
                        count += 1
                    
                    if count > 0:
                        avg_improvement = total_improvement / count
                        self.strategies['ensemble']['performance_with_news'] = avg_improvement
                        self.strategies['ensemble']['improvement'] = avg_improvement
                        self.strategies['ml_with_news']['performance_with_news'] = avg_improvement
                        self.strategies['ml_with_news']['improvement'] = avg_improvement
                
                # Обновляем показатели для технической стратегии
                if 'Technical' in results.get('results', {}):
                    tech_results = results['results']['Technical']
                    total_improvement = 0.0
                    count = 0
                    
                    for symbol, data in tech_results.items():
                        improvement = data.get('improvements', {}).get('return_improvement', 0.0)
                        total_improvement += improvement
                        count += 1
                    
                    if count > 0:
                        avg_improvement = total_improvement / count
                        self.strategies['technical']['performance_with_news'] = avg_improvement
                        self.strategies['technical']['improvement'] = avg_improvement
                        self.strategies['technical_with_news']['performance_with_news'] = avg_improvement
                        self.strategies['technical_with_news']['improvement'] = avg_improvement
                
                logger.info("✅ Показатели производительности обновлены из результатов тестирования")
        
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить показатели производительности: {e}")
    
    def get_strategy_info(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Получение информации о стратегии"""
        return self.strategies.get(strategy_id)
    
    def list_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Список всех доступных стратегий"""
        return self.strategies
    
    def get_best_strategies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Получение лучших стратегий по показателям"""
        strategies_list = []
        for strategy_id, info in self.strategies.items():
            strategies_list.append({
                'id': strategy_id,
                **info
            })
        
        # Сортируем по улучшению с новостями
        strategies_list.sort(key=lambda x: x['improvement'], reverse=True)
        return strategies_list[:limit]

class ConfigurableRobotStarter:
    """Конфигурируемый стартер для торговых роботов"""
    
    def __init__(self, config_file: str = "robot_config.json"):
        self.config_file = config_file
        self.config = {}
        self.strategy_registry = StrategyRegistry()
        self.news_analyzer = None
        self.news_manager = None
        self.trading_strategy = None
        self.telegram_notifier = None
        self.running = False
        
        # Загружаем конфигурацию
        self.load_configuration()
        
        # Инициализируем компоненты
        self.initialize_components()
        
        # Настройка обработки сигналов
        self.setup_signal_handlers()
    
    def load_configuration(self):
        """Загрузка конфигурации"""
        logger.info("🔧 Загрузка конфигурации робота...")
        
        # Загружаем .env файл
        env_paths = ['.env', 'config/.env', 'config/environments/.env']
        for path in env_paths:
            if os.path.exists(path):
                load_env_file(path)
                logger.info(f"✅ Конфигурация загружена из: {path}")
                break
        
        # Загружаем конфигурацию робота
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"✅ Конфигурация робота загружена из: {self.config_file}")
        else:
            # Создаем конфигурацию по умолчанию
            self.create_default_config()
            logger.info(f"✅ Создана конфигурация по умолчанию: {self.config_file}")
        
        # Проверяем обязательные переменные
        self.validate_configuration()
    
    def create_default_config(self):
        """Создание конфигурации по умолчанию"""
        self.config = {
            "robot": {
                "name": "Trading Robot",
                "version": "1.0.0",
                "description": "Конфигурируемый торговый робот"
            },
            "strategy": {
                "id": "ensemble",
                "name": "Ensemble Strategy",
                "use_news_analysis": True,
                "parameters": {
                    "confidence_threshold": 0.3,
                    "risk_per_trade": 0.02,
                    "max_positions": 5
                }
            },
            "instruments": {
                "symbols": ["SBER", "GAZP", "LKOH", "NVTK", "ROSN", "TATN"],
                "timeframe": "1min",
                "data_source": "tbank"
            },
            "news": {
                "enabled": True,
                "sources": ["russian_media", "moex", "telegram"],
                "update_interval": 300,  # 5 минут
                "sentiment_threshold": 0.2
            },
            "risk_management": {
                "max_drawdown": 0.15,
                "stop_loss": 0.05,
                "take_profit": 0.10,
                "position_sizing": "fixed"
            },
            "monitoring": {
                "telegram_notifications": True,
                "log_level": "INFO",
                "performance_check_interval": 3600,  # 1 час
                "retrain_interval": 86400  # 24 часа
            },
            "data": {
                "update_interval": 60,  # 1 минута
                "storage_path": "data/",
                "backup_enabled": True
            }
        }
        
        # Сохраняем конфигурацию
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def validate_configuration(self):
        """Валидация конфигурации"""
        required_vars = ['TBANK_TOKEN', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Отсутствуют обязательные переменные: {', '.join(missing_vars)}")
            sys.exit(1)
        
        # Проверяем стратегию
        strategy_id = self.config.get('strategy', {}).get('id', 'ensemble')
        if strategy_id not in self.strategy_registry.list_strategies():
            logger.error(f"❌ Неизвестная стратегия: {strategy_id}")
            sys.exit(1)
        
        logger.info("✅ Конфигурация валидна")
    
    def initialize_components(self):
        """Инициализация компонентов"""
        logger.info("🔧 Инициализация компонентов...")
        
        # Инициализируем анализатор новостей
        if self.config.get('news', {}).get('enabled', True):
            try:
                self.news_analyzer = RussianNewsAnalyzer("russian_news_config.json")
                self.news_manager = NewsDataManager()
                logger.info("✅ Анализатор новостей инициализирован")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка инициализации анализатора новостей: {e}")
        
        # Инициализируем торговую стратегию
        strategy_id = self.config.get('strategy', {}).get('id', 'ensemble')
        symbols = self.config.get('instruments', {}).get('symbols', ['SBER', 'GAZP'])
        
        try:
            self.trading_strategy = RussianTradingStrategy(symbols, "russian_news_config.json")
            logger.info(f"✅ Торговая стратегия '{strategy_id}' инициализирована")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка инициализации торговой стратегии: {e}")
        
        # Инициализируем уведомления
        if self.config.get('monitoring', {}).get('telegram_notifications', True):
            try:
                self.telegram_notifier = TradingNotifier()
                logger.info("✅ Уведомления Telegram инициализированы")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка инициализации уведомлений: {e}")
        
        logger.info("✅ Все компоненты инициализированы")
    
    def setup_signal_handlers(self):
        """Настройка обработки сигналов"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов"""
        logger.info(f"🛑 Получен сигнал {signum}, остановка робота...")
        self.running = False
    
    def start_robot(self):
        """Запуск робота"""
        logger.info("🚀 Запуск торгового робота...")
        
        strategy_info = self.strategy_registry.get_strategy_info(
            self.config.get('strategy', {}).get('id', 'ensemble')
        )
        
        if strategy_info:
            logger.info(f"📊 Стратегия: {strategy_info['name']}")
            logger.info(f"📈 Улучшение с новостями: {strategy_info['improvement']:+.2f}%")
        
        self.running = True
        
        try:
            # Запускаем основной цикл
            asyncio.run(self.main_loop())
        except KeyboardInterrupt:
            logger.info("🛑 Робот остановлен пользователем")
        except Exception as e:
            logger.error(f"❌ Ошибка в основном цикле: {e}")
        finally:
            self.cleanup()
    
    async def main_loop(self):
        """Основной цикл робота"""
        logger.info("🔄 Запуск основного цикла...")
        
        update_interval = self.config.get('data', {}).get('update_interval', 60)
        news_interval = self.config.get('news', {}).get('update_interval', 300)
        
        last_data_update = 0
        last_news_update = 0
        last_performance_check = 0
        
        while self.running:
            current_time = time.time()
            
            try:
                # Обновление данных
                if current_time - last_data_update >= update_interval:
                    await self.update_market_data()
                    last_data_update = current_time
                
                # Обновление новостей
                if current_time - last_news_update >= news_interval:
                    await self.update_news_data()
                    last_news_update = current_time
                
                # Проверка производительности
                performance_interval = self.config.get('monitoring', {}).get('performance_check_interval', 3600)
                if current_time - last_performance_check >= performance_interval:
                    await self.check_performance()
                    last_performance_check = current_time
                
                # Торговая логика
                await self.execute_trading_logic()
                
                # Небольшая пауза
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Ошибка в основном цикле: {e}")
                await asyncio.sleep(5)
    
    async def update_market_data(self):
        """Обновление рыночных данных"""
        logger.debug("📊 Обновление рыночных данных...")
        # Здесь будет логика обновления данных
        pass
    
    async def update_news_data(self):
        """Обновление данных новостей"""
        if not self.news_manager:
            return
        
        logger.debug("📰 Обновление данных новостей...")
        # Здесь будет логика обновления новостей
        pass
    
    async def check_performance(self):
        """Проверка производительности"""
        logger.debug("📈 Проверка производительности...")
        # Здесь будет логика проверки производительности
        pass
    
    async def execute_trading_logic(self):
        """Выполнение торговой логики"""
        logger.debug("💰 Выполнение торговой логики...")
        # Здесь будет основная торговая логика
        pass
    
    def cleanup(self):
        """Очистка ресурсов"""
        logger.info("🧹 Очистка ресурсов...")
        
        if self.news_analyzer:
            asyncio.run(self.news_analyzer.close())
        
        logger.info("✅ Очистка завершена")
    
    def show_strategies(self):
        """Показать список доступных стратегий"""
        print("\n📊 ДОСТУПНЫЕ СТРАТЕГИИ:")
        print("=" * 80)
        
        strategies = self.strategy_registry.list_strategies()
        
        for strategy_id, info in strategies.items():
            print(f"\n🔹 {info['name']} ({strategy_id})")
            print(f"   Тип: {info['type']}")
            print(f"   Описание: {info['description']}")
            print(f"   Без новостей: {info['performance_without_news']:+.2f}%")
            print(f"   С новостями: {info['performance_with_news']:+.2f}%")
            print(f"   Улучшение: {info['improvement']:+.2f}%")
        
        print(f"\n🏆 ТОП-5 ЛУЧШИХ СТРАТЕГИЙ:")
        print("-" * 80)
        
        best_strategies = self.strategy_registry.get_best_strategies(5)
        for i, strategy in enumerate(best_strategies, 1):
            print(f"{i}. {strategy['name']} - {strategy['improvement']:+.2f}% улучшение")
    
    def show_config(self):
        """Показать текущую конфигурацию"""
        print("\n⚙️ ТЕКУЩАЯ КОНФИГУРАЦИЯ:")
        print("=" * 80)
        print(json.dumps(self.config, ensure_ascii=False, indent=2))

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Конфигурируемый стартер торговых роботов')
    parser.add_argument('--config', '-c', default='robot_config.json', 
                       help='Файл конфигурации (по умолчанию: robot_config.json)')
    parser.add_argument('--strategy', '-s', help='ID стратегии для запуска')
    parser.add_argument('--symbols', nargs='+', help='Список символов для торговли')
    parser.add_argument('--list-strategies', action='store_true', 
                       help='Показать список доступных стратегий')
    parser.add_argument('--show-config', action='store_true', 
                       help='Показать текущую конфигурацию')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Запуск без реальной торговли')
    
    args = parser.parse_args()
    
    # Создаем стартер
    starter = ConfigurableRobotStarter(args.config)
    
    # Показываем стратегии если запрошено
    if args.list_strategies:
        starter.show_strategies()
        return
    
    # Показываем конфигурацию если запрошено
    if args.show_config:
        starter.show_config()
        return
    
    # Обновляем конфигурацию из аргументов
    if args.strategy:
        starter.config['strategy']['id'] = args.strategy
    
    if args.symbols:
        starter.config['instruments']['symbols'] = args.symbols
    
    if args.dry_run:
        starter.config['robot']['dry_run'] = True
        logger.info("🧪 Режим тестирования (без реальной торговли)")
    
    # Запускаем робота
    starter.start_robot()

if __name__ == "__main__":
    main()
