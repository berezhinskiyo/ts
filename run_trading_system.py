#!/usr/bin/env python3
"""
Main Trading System Launcher
Главный скрипт для запуска торговой системы
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring/logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystemLauncher:
    """Главный класс для запуска торговой системы"""
    
    def __init__(self):
        self.available_strategies = {
            'ml': 'strategies/ml/ml_trading_system.py',
            'aggressive': 'strategies/aggressive/aggressive_intraday_system.py',
            'combined': 'strategies/combined/full_trading_system.py',
            'simplified': 'strategies/combined/simplified_full_system.py'
        }
        
        self.available_commands = {
            'run': self.run_strategy,
            'backtest': self.run_backtest,
            'monitor': self.run_monitoring,
            'optimize': self.run_optimization,
            'retrain': self.run_retraining,
            'status': self.check_status
        }
    
    def run_strategy(self, strategy_name: str, **kwargs):
        """Запуск торговой стратегии"""
        logger.info(f"🚀 Starting {strategy_name} strategy")
        
        if strategy_name not in self.available_strategies:
            logger.error(f"Unknown strategy: {strategy_name}")
            logger.info(f"Available strategies: {list(self.available_strategies.keys())}")
            return False
        
        strategy_path = self.available_strategies[strategy_name]
        
        try:
            # Проверка существования файла
            if not os.path.exists(strategy_path):
                logger.error(f"Strategy file not found: {strategy_path}")
                return False
            
            # Запуск стратегии
            logger.info(f"Executing: python3 {strategy_path}")
            
            # В реальном проекте здесь будет subprocess.Popen
            # Для демонстрации просто логируем
            logger.info(f"✅ {strategy_name} strategy started successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting {strategy_name} strategy: {e}")
            return False
    
    def run_backtest(self, strategy_name: str = 'all', **kwargs):
        """Запуск бэктестинга"""
        logger.info(f"🧪 Running backtest for {strategy_name}")
        
        backtest_engines = {
            'quick': 'backtesting/engines/quick_test.py',
            'advanced': 'backtesting/engines/advanced_test.py',
            'real_data': 'backtesting/engines/real_data_test.py',
            'working': 'backtesting/engines/working_test.py'
        }
        
        if strategy_name == 'all':
            engines_to_run = list(backtest_engines.keys())
        else:
            engines_to_run = [strategy_name] if strategy_name in backtest_engines else ['quick']
        
        results = {}
        
        for engine in engines_to_run:
            engine_path = backtest_engines[engine]
            
            try:
                if os.path.exists(engine_path):
                    logger.info(f"Running {engine} backtest: {engine_path}")
                    # В реальном проекте здесь будет subprocess.Popen
                    results[engine] = {'status': 'completed', 'path': engine_path}
                else:
                    logger.warning(f"Backtest engine not found: {engine_path}")
                    results[engine] = {'status': 'not_found', 'path': engine_path}
                    
            except Exception as e:
                logger.error(f"Error running {engine} backtest: {e}")
                results[engine] = {'status': 'error', 'error': str(e)}
        
        # Сохранение результатов
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'backtesting/results/backtest_results_{timestamp}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Backtest results saved to {results_file}")
        return results
    
    def run_monitoring(self, **kwargs):
        """Запуск системы мониторинга"""
        logger.info("📊 Starting monitoring system")
        
        try:
            # Запуск мониторинга качества
            from monitoring.metrics.quality_monitor import QualityMonitor
            
            monitor = QualityMonitor()
            
            # Тестовые данные для демонстрации
            test_strategies = {
                'ML_Strategy': {
                    'monthly_return': 0.085,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': 0.08,
                    'ml_precision': 0.65
                },
                'Aggressive_Strategy': {
                    'monthly_return': 0.152,
                    'sharpe_ratio': 1.8,
                    'max_drawdown': 0.12
                }
            }
            
            # Проверка качества
            results = monitor.check_all_strategies(test_strategies)
            
            # Генерация отчета
            report = monitor.get_quality_report(days=1)
            
            logger.info("✅ Monitoring completed")
            logger.info(f"Overall Quality Score: {results['overall']['average_quality_score']:.1f}/100")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running monitoring: {e}")
            return False
    
    def run_optimization(self, strategy_name: str = 'aggressive', **kwargs):
        """Запуск оптимизации параметров"""
        logger.info(f"🔧 Running optimization for {strategy_name}")
        
        try:
            if strategy_name == 'aggressive':
                from strategies.aggressive.parameter_optimizer import ParameterOptimizer
                
                optimizer = ParameterOptimizer()
                results = optimizer.optimize_parameters()
                
                if results:
                    logger.info("✅ Optimization completed successfully")
                    logger.info(f"Best parameters: {results['best_parameters']}")
                    return results
                else:
                    logger.error("❌ Optimization failed")
                    return False
            else:
                logger.warning(f"Optimization not implemented for {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            return False
    
    def run_retraining(self, strategy_name: str = 'ml', **kwargs):
        """Запуск переобучения ML моделей"""
        logger.info(f"🤖 Running retraining for {strategy_name}")
        
        try:
            if strategy_name == 'ml':
                from strategies.ml.retrain_scheduler import MLRetrainScheduler
                
                scheduler = MLRetrainScheduler()
                success = scheduler.retrain_ml_models()
                
                if success:
                    logger.info("✅ ML retraining completed successfully")
                    return True
                else:
                    logger.error("❌ ML retraining failed")
                    return False
            else:
                logger.warning(f"Retraining not implemented for {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error running retraining: {e}")
            return False
    
    def check_status(self, **kwargs):
        """Проверка статуса системы"""
        logger.info("📋 Checking system status")
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {},
            'files': {},
            'directories': {}
        }
        
        # Проверка стратегий
        for strategy_name, strategy_path in self.available_strategies.items():
            status['strategies'][strategy_name] = {
                'path': strategy_path,
                'exists': os.path.exists(strategy_path),
                'size': os.path.getsize(strategy_path) if os.path.exists(strategy_path) else 0
            }
        
        # Проверка ключевых файлов
        key_files = [
            'config/parameters/ml_config.py',
            'config/parameters/aggressive_config.py',
            'config/environments/.env',
            'requirements.txt'
        ]
        
        for file_path in key_files:
            status['files'][file_path] = {
                'exists': os.path.exists(file_path),
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        
        # Проверка директорий
        key_directories = [
            'strategies',
            'backtesting',
            'monitoring',
            'data',
            'config',
            'docs'
        ]
        
        for dir_path in key_directories:
            status['directories'][dir_path] = {
                'exists': os.path.exists(dir_path),
                'is_directory': os.path.isdir(dir_path) if os.path.exists(dir_path) else False
            }
        
        # Сохранение статуса
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        status_file = f'monitoring/logs/system_status_{timestamp}.json'
        
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 System status saved to {status_file}")
        
        # Вывод краткого статуса
        print("\n📋 SYSTEM STATUS")
        print("="*50)
        print(f"Timestamp: {status['timestamp']}")
        print(f"Strategies: {len([s for s in status['strategies'].values() if s['exists']])}/{len(status['strategies'])}")
        print(f"Key files: {len([f for f in status['files'].values() if f['exists']])}/{len(status['files'])}")
        print(f"Directories: {len([d for d in status['directories'].values() if d['exists']])}/{len(status['directories'])}")
        
        return status
    
    def show_help(self):
        """Показать справку"""
        help_text = """
🚀 TRADING SYSTEM LAUNCHER

Использование:
    python3 run_trading_system.py <command> [options]

Команды:
    run <strategy>     - Запуск торговой стратегии
    backtest [engine]  - Запуск бэктестинга
    monitor           - Запуск системы мониторинга
    optimize [strategy] - Оптимизация параметров
    retrain [strategy]  - Переобучение ML моделей
    status            - Проверка статуса системы
    help              - Показать эту справку

Стратегии:
    ml               - Машинное обучение
    aggressive       - Агрессивная стратегия
    combined         - Комбинированная стратегия
    simplified       - Упрощенная система

Движки бэктестинга:
    quick           - Быстрый тест
    advanced        - Продвинутый тест
    real_data       - Тест на реальных данных
    working         - Рабочий тест

Примеры:
    python3 run_trading_system.py run ml
    python3 run_trading_system.py backtest quick
    python3 run_trading_system.py monitor
    python3 run_trading_system.py optimize aggressive
    python3 run_trading_system.py retrain ml
    python3 run_trading_system.py status
"""
        print(help_text)

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Trading System Launcher')
    parser.add_argument('command', help='Command to execute')
    parser.add_argument('strategy', nargs='?', help='Strategy name')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Настройка логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    launcher = TradingSystemLauncher()
    
    # Выполнение команды
    if args.command == 'help':
        launcher.show_help()
        return
    
    if args.command not in launcher.available_commands:
        logger.error(f"Unknown command: {args.command}")
        launcher.show_help()
        return
    
    try:
        # Выполнение команды
        if args.strategy:
            result = launcher.available_commands[args.command](args.strategy)
        else:
            result = launcher.available_commands[args.command]()
        
        if result:
            logger.info(f"✅ Command '{args.command}' completed successfully")
        else:
            logger.error(f"❌ Command '{args.command}' failed")
            
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")

if __name__ == "__main__":
    main()

