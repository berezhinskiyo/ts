# 📁 РУКОВОДСТВО ПО ИСПОЛЬЗОВАНИЮ .ENV ФАЙЛОВ

## 📋 Обзор

Система поддерживает различные способы указания .env файлов при запуске скриптов. Это позволяет гибко настраивать конфигурацию для разных окружений.

## 🚀 Способы запуска

### **1. Стандартный запуск (автопоиск)**

```bash
python test_telegram_notifications.py
```

Система автоматически ищет .env файл в следующих местах:

- `.env` (текущая директория)
- `config/.env`
- `config/environments/.env`
- `config/parameters/.env`
- `~/.trading/.env`
- `/etc/trading/.env`

### **2. Указание конкретного пути**

```bash
python test_telegram_notifications.py --env-file config/environments/.env
```

### **3. Прямое указание токенов**

```bash
python test_telegram_notifications.py --bot-token YOUR_BOT_TOKEN --chat-id YOUR_CHAT_ID
```

### **4. Комбинированный способ**

```bash
python test_telegram_notifications.py --env-file config/environments/.env --bot-token YOUR_BOT_TOKEN
```

## 📁 Структура директорий

### **Рекомендуемая структура**

```
project/
├── .env                          # Основной .env файл
├── config/
│   ├── .env                      # Конфигурация для config
│   ├── environments/
│   │   ├── .env                  # Окружение по умолчанию
│   │   ├── .env.production       # Продакшн окружение
│   │   ├── .env.development      # Разработка
│   │   └── .env.testing          # Тестирование
│   └── parameters/
│       └── .env                  # Параметры стратегий
├── test_telegram_notifications.py
└── env_loader.py
```

### **Примеры .env файлов**

#### **Основной .env файл**

```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789

# T-Bank API Configuration
TBANK_TOKEN=your_tbank_token_here
TBANK_SANDBOX_TOKEN=your_sandbox_token_here

# Trading Configuration
INITIAL_CAPITAL=100000
MAX_RISK_PER_TRADE=0.02
USE_SANDBOX=true
```

#### **Продакшн окружение (.env.production)**

```env
# Production Telegram Bot
TELEGRAM_BOT_TOKEN=987654321:ZYXwvuTSRqpoNMLkjihGFEdcba
TELEGRAM_CHAT_ID=987654321

# Production T-Bank API
TBANK_TOKEN=production_tbank_token
TBANK_SANDBOX_TOKEN=
USE_SANDBOX=false

# Production Trading Settings
INITIAL_CAPITAL=1000000
MAX_RISK_PER_TRADE=0.01
```

#### **Тестовое окружение (.env.testing)**

```env
# Test Telegram Bot
TELEGRAM_BOT_TOKEN=111222333:TestBotTokenForTesting
TELEGRAM_CHAT_ID=111222333

# Test T-Bank API
TBANK_TOKEN=test_tbank_token
TBANK_SANDBOX_TOKEN=test_sandbox_token
USE_SANDBOX=true

# Test Trading Settings
INITIAL_CAPITAL=10000
MAX_RISK_PER_TRADE=0.05
```

## 🔧 Утилиты для работы с .env

### **env_loader.py - Универсальный загрузчик**

#### **Поиск всех .env файлов**

```bash
python env_loader.py --list
```

#### **Загрузка конкретного файла**

```bash
python env_loader.py --env-file config/environments/.env.production
```

#### **Тихий режим**

```bash
python env_loader.py --env-file config/.env --quiet
```

### **Использование в коде**

```python
from env_loader import load_env_file

# Автопоиск .env файла
success = load_env_file()

# Загрузка конкретного файла
success = load_env_file('config/environments/.env.production')

# Тихий режим
success = load_env_file(verbose=False)
```

## 🚀 Примеры запуска для разных сценариев

### **Разработка**

```bash
# Использует .env в корне проекта
python test_telegram_notifications.py
```

### **Тестирование**

```bash
# Использует тестовое окружение
python test_telegram_notifications.py --env-file config/environments/.env.testing
```

### **Продакшн**

```bash
# Использует продакшн окружение
python test_telegram_notifications.py --env-file config/environments/.env.production
```

### **Быстрое тестирование с токенами**

```bash
# Не требует .env файла
python test_telegram_notifications.py \
  --bot-token "123456789:ABCdefGHIjklMNOpqrsTUVwxyz" \
  --chat-id "123456789"
```

### **Переопределение токенов**

```bash
# Загружает .env файл, но переопределяет токены
python test_telegram_notifications.py \
  --env-file config/environments/.env \
  --bot-token "999888777:OverrideToken" \
  --chat-id "999888777"
```

## 🔍 Отладка и диагностика

### **Проверка найденных .env файлов**

```bash
python env_loader.py --list
```

### **Проверка загрузки переменных**

```python
import os
from env_loader import load_env_file

# Загружаем .env файл
load_env_file('config/environments/.env')

# Проверяем переменные
print(f"Bot Token: {os.getenv('TELEGRAM_BOT_TOKEN', 'НЕ НАЙДЕН')}")
print(f"Chat ID: {os.getenv('TELEGRAM_CHAT_ID', 'НЕ НАЙДЕН')}")
```

### **Логирование загрузки**

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from env_loader import load_env_file
load_env_file('config/environments/.env')
```

## 🛠️ Интеграция в другие скрипты

### **В торговых стратегиях**

```python
#!/usr/bin/env python3
import sys
import os

# Добавляем путь к модулям
sys.path.append('.')

# Загружаем .env файл
from env_loader import load_env_file
load_env_file('config/environments/.env')

# Теперь можно использовать переменные окружения
from config import Config
```

### **В скриптах бэктестинга**

```python
#!/usr/bin/env python3
import argparse
from env_loader import load_env_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', default='.env')
    args = parser.parse_args()

    # Загружаем .env файл
    load_env_file(args.env_file)

    # Запускаем бэктестинг
    # ...

if __name__ == "__main__":
    main()
```

## 🔒 Безопасность

### **Рекомендации по безопасности**

1. **Не коммитьте .env файлы** в git
2. **Используйте разные токены** для разных окружений
3. **Ограничьте права доступа** к .env файлам
4. **Используйте переменные окружения** в продакшне

### **.gitignore**

```gitignore
# Environment files
.env
.env.*
config/environments/.env*
config/parameters/.env*
```

### **Права доступа**

```bash
# Ограничить доступ к .env файлам
chmod 600 config/environments/.env*
chmod 600 config/parameters/.env*
```

## 📚 Дополнительные возможности

### **Переменные окружения системы**

```bash
# Установка переменных в системе
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="123456789"

# Запуск без .env файла
python test_telegram_notifications.py
```

### **Docker поддержка**

```dockerfile
# Dockerfile
FROM python:3.9

COPY . /app
WORKDIR /app

# Копируем .env файл
COPY config/environments/.env /app/.env

RUN pip install -r requirements.txt
CMD ["python", "test_telegram_notifications.py"]
```

### **Docker Compose**

```yaml
# docker-compose.yml
version: "3.8"
services:
  trading-bot:
    build: .
    env_file:
      - config/environments/.env.production
    volumes:
      - ./data:/app/data
```

## 🎯 Заключение

Система поддерживает гибкую настройку .env файлов для различных сценариев использования:

### **✅ Преимущества**

- **Автопоиск** .env файлов в стандартных местах
- **Гибкое указание** путей через аргументы
- **Переопределение** переменных из командной строки
- **Поддержка** различных окружений
- **Безопасность** и изоляция конфигураций

### **🚀 Готово к использованию**

- **Простая настройка** для разработки
- **Гибкая конфигурация** для продакшна
- **Удобные утилиты** для диагностики
- **Подробная документация** и примеры

---

**Дата создания**: 15 сентября 2025  
**Версия**: 1.0  
**Статус**: ✅ Готово к использованию

