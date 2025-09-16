#!/usr/bin/env python3
"""
Универсальный загрузчик .env файлов для торговых стратегий
"""

import os
import sys
import argparse
from pathlib import Path

def load_env_file(env_path: str = None, verbose: bool = True) -> bool:
    """
    Загружает .env файл с поддержкой различных путей
    
    Args:
        env_path: Путь к .env файлу
        verbose: Выводить ли информацию о загрузке
    
    Returns:
        bool: True если файл загружен успешно, False иначе
    """
    
    # Возможные пути к .env файлу
    possible_paths = []
    
    if env_path:
        # Если указан конкретный путь
        possible_paths.append(env_path)
    else:
        # Стандартные пути
        possible_paths.extend([
            '.env',                                    # Текущая директория
            'config/.env',                            # config директория
            'config/environments/.env',               # config/environments
            'config/parameters/.env',                 # config/parameters
            os.path.expanduser('~/.trading/.env'),    # Домашняя директория
            '/etc/trading/.env'                       # Системная директория
        ])
    
    # Ищем существующий .env файл
    env_file = None
    for path in possible_paths:
        if os.path.exists(path):
            env_file = path
            break
    
    if not env_file:
        if verbose:
            print("❌ .env файл не найден в следующих местах:")
            for path in possible_paths:
                print(f"   - {path}")
            print("\n📝 Создайте .env файл в одном из указанных мест")
        return False
    
    # Загружаем .env файл
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        
        if verbose:
            print(f"✅ Переменные окружения загружены из: {env_file}")
            
            # Проверяем основные переменные
            required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
            missing_vars = []
            
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"⚠️ Отсутствуют переменные: {', '.join(missing_vars)}")
            else:
                print("✅ Все необходимые переменные найдены")
        
        return True
        
    except ImportError:
        if verbose:
            print("❌ python-dotenv не установлен")
            print("📦 Установите: pip install python-dotenv")
        return False
    except Exception as e:
        if verbose:
            print(f"❌ Ошибка загрузки .env файла: {e}")
        return False

def get_env_paths() -> list:
    """Возвращает список возможных путей к .env файлам"""
    return [
        '.env',
        'config/.env',
        'config/environments/.env',
        'config/parameters/.env',
        os.path.expanduser('~/.trading/.env'),
        '/etc/trading/.env'
    ]

def find_env_files() -> list:
    """Находит все существующие .env файлы"""
    existing_files = []
    for path in get_env_paths():
        if os.path.exists(path):
            existing_files.append(path)
    return existing_files

def print_env_status():
    """Выводит статус .env файлов"""
    print("📁 ПОИСК .ENV ФАЙЛОВ")
    print("=" * 30)
    
    existing_files = find_env_files()
    
    if existing_files:
        print("✅ Найденные .env файлы:")
        for file_path in existing_files:
            print(f"   - {file_path}")
    else:
        print("❌ .env файлы не найдены")
        print("\n📝 Возможные места для размещения .env файла:")
        for path in get_env_paths():
            print(f"   - {path}")
    
    print()

def main():
    """Основная функция для командной строки"""
    parser = argparse.ArgumentParser(description='Загрузчик .env файлов для торговых стратегий')
    parser.add_argument(
        '--env-file', 
        type=str,
        help='Путь к .env файлу'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='Показать все найденные .env файлы'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Тихий режим (минимальный вывод)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print_env_status()
        return
    
    # Загружаем .env файл
    success = load_env_file(args.env_file, verbose=not args.quiet)
    
    if not success and not args.quiet:
        print("\n💡 Использование:")
        print("   python env_loader.py --env-file path/to/.env")
        print("   python env_loader.py --list")
        print("   python env_loader.py --quiet")

if __name__ == "__main__":
    main()

