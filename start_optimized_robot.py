#!/usr/bin/env python3
"""
Скрипт запуска оптимизированного торгового робота
с наиболее прибыльными инструментами и стратегиями
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Проверка требований для запуска"""
    logger.info("🔍 Проверка требований...")
    
    # Проверяем наличие конфигурационных файлов
    required_files = [
        'configurable_robot_starter.py',
        'robot_config_optimized.json',
        'data_update_service.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Отсутствуют файлы: {', '.join(missing_files)}")
        return False
    
    # Проверяем наличие .env файла
    env_files = [
        'config/environments/.env',
        'config/.env',
        '.env'
    ]
    
    env_found = False
    for env_file in env_files:
        if os.path.exists(env_file):
            logger.info(f"✅ Найден .env файл: {env_file}")
            env_found = True
            break
    
    if not env_found:
        logger.warning("⚠️ .env файл не найден, будут использованы системные переменные")
    
    logger.info("✅ Все требования выполнены")
    return True

def start_data_service():
    """Запуск сервиса обновления данных"""
    logger.info("🚀 Запуск сервиса обновления данных...")
    
    try:
        # Запускаем сервис обновления данных в фоне
        process = subprocess.Popen([
            sys.executable, 'data_update_service.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"✅ Сервис обновления данных запущен (PID: {process.pid})")
        return process
        
    except Exception as e:
        logger.error(f"❌ Ошибка запуска сервиса обновления данных: {e}")
        return None

def start_trading_robot():
    """Запуск торгового робота"""
    logger.info("🤖 Запуск торгового робота...")
    
    try:
        # Запускаем торговый робот с оптимизированной конфигурацией
        cmd = [
            sys.executable, 'configurable_robot_starter.py',
            '--config', 'robot_config_optimized.json',
            '--strategy', 'ensemble',
            '--symbols', 'PIKK', 'SGZH', 'SBER'
        ]
        
        logger.info(f"📋 Команда запуска: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd)
        
        logger.info(f"✅ Торговый робот запущен (PID: {process.pid})")
        return process
        
    except Exception as e:
        logger.error(f"❌ Ошибка запуска торгового робота: {e}")
        return None

def show_optimization_info():
    """Показать информацию об оптимизации"""
    print("\n" + "="*80)
    print("🚀 ОПТИМИЗИРОВАННЫЙ ТОРГОВЫЙ РОБОТ")
    print("="*80)
    
    print("\n📊 ВЫБРАННЫЕ ИНСТРУМЕНТЫ:")
    print("  🏆 PIKK - +46.66% улучшение с новостями")
    print("  ✅ SGZH - +16.04% улучшение с новостями")
    print("  ✅ SBER - +2.30% улучшение с новостями")
    
    print("\n🤖 СТРАТЕГИЯ:")
    print("  📈 Ensemble Strategy - +13.00% среднее улучшение")
    print("  📰 Анализ новостей включен")
    print("  🎯 Агрессивный режим")
    
    print("\n⚙️ НАСТРОЙКИ:")
    print("  💰 Максимальная просадка: 12%")
    print("  🛑 Stop Loss: 4%")
    print("  🎯 Take Profit: 8%")
    print("  📊 Максимум позиций: 3")
    
    print("\n📱 МОНИТОРИНГ:")
    print("  📲 Telegram уведомления включены")
    print("  ⏰ Проверка производительности каждые 30 минут")
    print("  📊 Ежедневные отчеты")
    print("  🚨 Алерты при ошибках")
    
    print("\n⏰ РАСПИСАНИЕ ТОРГОВЛИ:")
    print("  🕘 Начало: 09:00 МСК")
    print("  🕕 Окончание: 18:45 МСК")
    print("  📅 Только рабочие дни")
    
    print("\n" + "="*80)

def monitor_processes(data_process, robot_process):
    """Мониторинг процессов"""
    logger.info("👀 Начало мониторинга процессов...")
    
    try:
        while True:
            # Проверяем статус сервиса обновления данных
            if data_process and data_process.poll() is not None:
                logger.error("❌ Сервис обновления данных остановлен")
                break
            
            # Проверяем статус торгового робота
            if robot_process and robot_process.poll() is not None:
                logger.error("❌ Торговый робот остановлен")
                break
            
            # Ждем 30 секунд перед следующей проверкой
            time.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("🛑 Получен сигнал остановки...")
        
        # Останавливаем процессы
        if data_process:
            data_process.terminate()
            logger.info("🛑 Сервис обновления данных остановлен")
        
        if robot_process:
            robot_process.terminate()
            logger.info("🛑 Торговый робот остановлен")

def main():
    """Основная функция"""
    print("🚀 ЗАПУСК ОПТИМИЗИРОВАННОГО ТОРГОВОГО РОБОТА")
    print("="*60)
    
    # Показываем информацию об оптимизации
    show_optimization_info()
    
    # Проверяем требования
    if not check_requirements():
        logger.error("❌ Требования не выполнены, завершение работы")
        sys.exit(1)
    
    # Запускаем сервис обновления данных
    data_process = start_data_service()
    if not data_process:
        logger.error("❌ Не удалось запустить сервис обновления данных")
        sys.exit(1)
    
    # Ждем немного для инициализации сервиса
    logger.info("⏳ Ожидание инициализации сервиса обновления данных...")
    time.sleep(5)
    
    # Запускаем торговый робот
    robot_process = start_trading_robot()
    if not robot_process:
        logger.error("❌ Не удалось запустить торговый робот")
        if data_process:
            data_process.terminate()
        sys.exit(1)
    
    logger.info("✅ Все сервисы запущены успешно!")
    logger.info("📱 Мониторинг в Telegram активен")
    logger.info("🛑 Для остановки нажмите Ctrl+C")
    
    # Мониторим процессы
    monitor_processes(data_process, robot_process)
    
    logger.info("👋 Работа завершена")

if __name__ == "__main__":
    main()
