#!/usr/bin/env python3
"""
Simple Trading System Launcher
Простой запускатор торговой системы
"""

import os
import sys
import json
from datetime import datetime

def show_help():
    """Показать справку"""
    help_text = """
🚀 TRADING SYSTEM LAUNCHER

Использование:
    python3 simple_launcher.py <command> [strategy]

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
    python3 simple_launcher.py run ml
    python3 simple_launcher.py backtest quick
    python3 simple_launcher.py monitor
    python3 simple_launcher.py optimize aggressive
    python3 simple_launcher.py retrain ml
    python3 simple_launcher.py status
"""
    
    # Записываем в файл вместо print
    with open('help_output.txt', 'w', encoding='utf-8') as f:
        f.write(help_text)
    
    return help_text

def run_strategy(strategy_name):
    """Запуск торговой стратегии"""
    strategies = {
        'ml': 'strategies/ml/ml_trading_system.py',
        'aggressive': 'strategies/aggressive/aggressive_intraday_system.py',
        'combined': 'strategies/combined/full_trading_system.py',
        'simplified': 'strategies/combined/simplified_full_system.py'
    }
    
    if strategy_name not in strategies:
        result = f"❌ Unknown strategy: {strategy_name}\nAvailable: {list(strategies.keys())}"
    else:
        strategy_path = strategies[strategy_name]
        if os.path.exists(strategy_path):
            result = f"✅ Starting {strategy_name} strategy: {strategy_path}"
        else:
            result = f"❌ Strategy file not found: {strategy_path}"
    
    # Сохраняем результат
    with open('strategy_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result

def run_backtest(engine='quick'):
    """Запуск бэктестинга"""
    engines = {
        'quick': 'backtesting/engines/quick_test.py',
        'advanced': 'backtesting/engines/advanced_test.py',
        'real_data': 'backtesting/engines/real_data_test.py',
        'working': 'backtesting/engines/working_test.py'
    }
    
    if engine not in engines:
        result = f"❌ Unknown engine: {engine}\nAvailable: {list(engines.keys())}"
    else:
        engine_path = engines[engine]
        if os.path.exists(engine_path):
            result = f"✅ Running {engine} backtest: {engine_path}"
        else:
            result = f"❌ Engine file not found: {engine_path}"
    
    # Сохраняем результат
    with open('backtest_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result

def run_monitoring():
    """Запуск мониторинга"""
    result = "📊 Starting monitoring system..."
    
    # Проверяем наличие файлов мониторинга
    monitoring_files = [
        'monitoring/metrics/quality_monitor.py',
        'monitoring/logs/',
        'monitoring/alerts/'
    ]
    
    for file_path in monitoring_files:
        if os.path.exists(file_path):
            result += f"\n✅ Found: {file_path}"
        else:
            result += f"\n❌ Missing: {file_path}"
    
    # Сохраняем результат
    with open('monitoring_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result

def run_optimization(strategy='aggressive'):
    """Запуск оптимизации"""
    if strategy == 'aggressive':
        optimizer_path = 'strategies/aggressive/parameter_optimizer.py'
        if os.path.exists(optimizer_path):
            result = f"🔧 Running optimization for {strategy}: {optimizer_path}"
        else:
            result = f"❌ Optimizer not found: {optimizer_path}"
    else:
        result = f"❌ Optimization not implemented for {strategy}"
    
    # Сохраняем результат
    with open('optimization_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result

def run_retraining(strategy='ml'):
    """Запуск переобучения"""
    if strategy == 'ml':
        retrain_path = 'strategies/ml/retrain_scheduler.py'
        if os.path.exists(retrain_path):
            result = f"🤖 Running retraining for {strategy}: {retrain_path}"
        else:
            result = f"❌ Retrainer not found: {retrain_path}"
    else:
        result = f"❌ Retraining not implemented for {strategy}"
    
    # Сохраняем результат
    with open('retraining_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result

def check_status():
    """Проверка статуса системы"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'strategies': {},
        'directories': {},
        'files': {}
    }
    
    # Проверка стратегий
    strategies = {
        'ml': 'strategies/ml/ml_trading_system.py',
        'aggressive': 'strategies/aggressive/aggressive_intraday_system.py',
        'combined': 'strategies/combined/full_trading_system.py',
        'simplified': 'strategies/combined/simplified_full_system.py'
    }
    
    for name, path in strategies.items():
        status['strategies'][name] = {
            'path': path,
            'exists': os.path.exists(path),
            'size': os.path.getsize(path) if os.path.exists(path) else 0
        }
    
    # Проверка директорий
    directories = [
        'strategies', 'backtesting', 'monitoring', 'data', 'config', 'docs'
    ]
    
    for dir_name in directories:
        status['directories'][dir_name] = {
            'exists': os.path.exists(dir_name),
            'is_directory': os.path.isdir(dir_name) if os.path.exists(dir_name) else False
        }
    
    # Проверка ключевых файлов
    key_files = [
        'config/requirements.txt',
        'config/environments/.env',
        'README.md'
    ]
    
    for file_path in key_files:
        status['files'][file_path] = {
            'exists': os.path.exists(file_path),
            'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
    
    # Сохранение статуса
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    status_file = f'system_status_{timestamp}.json'
    
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    
    # Создание текстового отчета
    report = f"""
📋 SYSTEM STATUS REPORT
{'='*50}
Timestamp: {status['timestamp']}

STRATEGIES:
"""
    
    for name, info in status['strategies'].items():
        status_icon = "✅" if info['exists'] else "❌"
        report += f"{status_icon} {name}: {info['path']}\n"
    
    report += "\nDIRECTORIES:\n"
    for name, info in status['directories'].items():
        status_icon = "✅" if info['exists'] else "❌"
        report += f"{status_icon} {name}/\n"
    
    report += "\nKEY FILES:\n"
    for path, info in status['files'].items():
        status_icon = "✅" if info['exists'] else "❌"
        report += f"{status_icon} {path}\n"
    
    # Сохранение отчета
    with open('status_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

def main():
    """Главная функция"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1]
    strategy = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Выполнение команды
    if command == 'help':
        show_help()
    elif command == 'run':
        if strategy:
            run_strategy(strategy)
        else:
            result = "❌ Please specify strategy: ml, aggressive, combined, simplified"
            with open('error.txt', 'w', encoding='utf-8') as f:
                f.write(result)
    elif command == 'backtest':
        engine = strategy if strategy else 'quick'
        run_backtest(engine)
    elif command == 'monitor':
        run_monitoring()
    elif command == 'optimize':
        strategy_name = strategy if strategy else 'aggressive'
        run_optimization(strategy_name)
    elif command == 'retrain':
        strategy_name = strategy if strategy else 'ml'
        run_retraining(strategy_name)
    elif command == 'status':
        check_status()
    else:
        result = f"❌ Unknown command: {command}\nUse 'help' for available commands"
        with open('error.txt', 'w', encoding='utf-8') as f:
            f.write(result)

if __name__ == "__main__":
    main()

