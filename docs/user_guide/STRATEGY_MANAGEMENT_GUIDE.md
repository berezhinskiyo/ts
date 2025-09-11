# 🚀 РУКОВОДСТВО ПО УПРАВЛЕНИЮ ТОРГОВЫМИ СТРАТЕГИЯМИ

## 📁 СТРУКТУРА ПРОЕКТА

```
trading_system/
├── strategies/                 # Торговые стратегии
│   ├── ml/                    # Машинное обучение
│   │   └── ml_trading_system.py
│   ├── aggressive/            # Агрессивные стратегии
│   │   ├── aggressive_intraday_system.py
│   │   └── intraday_trading_system.py
│   ├── options/               # Опционные стратегии
│   └── combined/              # Комбинированные стратегии
│       ├── full_trading_system.py
│       └── simplified_full_system.py
├── backtesting/               # Тестирование стратегий
│   ├── engines/              # Движки бэктестинга
│   └── results/              # Результаты тестов
├── data/                     # Данные
│   ├── historical/           # Исторические данные
│   └── real_time/            # Данные в реальном времени
├── monitoring/               # Мониторинг
│   ├── logs/                 # Логи
│   ├── metrics/              # Метрики
│   └── alerts/               # Уведомления
├── config/                   # Конфигурация
│   ├── environments/         # Окружения (.env)
│   └── parameters/           # Параметры стратегий
├── docs/                     # Документация
│   ├── user_guide/           # Руководства пользователя
│   ├── api/                  # API документация
│   └── deployment/           # Развертывание
└── utils/                    # Утилиты
    ├── helpers/              # Вспомогательные функции
    └── validators/           # Валидаторы
```

---

## 🚀 ЗАПУСК СТРАТЕГИЙ

### 1. 🤖 ML Strategy (Машинное обучение)

#### Запуск:

```bash
cd strategies/ml
python3 ml_trading_system.py
```

#### Параметры:

```python
# config/parameters/ml_config.py
ML_CONFIG = {
    'models': ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM'],
    'features': 50,
    'confidence_threshold': 0.7,
    'retrain_frequency': 'daily',
    'lookback_period': 252
}
```

#### Мониторинг:

```bash
# Просмотр логов
tail -f monitoring/logs/ml_strategy.log

# Метрики производительности
python3 monitoring/metrics/ml_metrics.py
```

### 2. ⚡ Aggressive Strategy (Агрессивная стратегия)

#### Запуск:

```bash
cd strategies/aggressive
python3 aggressive_intraday_system.py
```

#### Параметры:

```python
# config/parameters/aggressive_config.py
AGGRESSIVE_CONFIG = {
    'leverage': 20.0,
    'max_position_size': 0.1,
    'profit_target': 0.02,
    'stop_loss': 0.01,
    'trend_threshold': 0.02,
    'volatility_threshold': 0.03
}
```

#### Мониторинг:

```bash
# Мониторинг рисков
python3 monitoring/alerts/risk_monitor.py

# Отслеживание плеча
python3 monitoring/metrics/leverage_tracker.py
```

### 3. 🎯 Combined Strategy (Комбинированная стратегия)

#### Запуск:

```bash
cd strategies/combined
python3 full_trading_system.py
```

#### Параметры:

```python
# config/parameters/combined_config.py
COMBINED_CONFIG = {
    'strategy_weights': {
        'ML_Strategy': 0.3,
        'Intraday_Strategy': 0.25,
        'Aggressive_Strategy': 0.25,
        'Options_Strategy': 0.2
    },
    'rebalance_frequency': 'daily',
    'risk_limit': 0.05
}
```

---

## 📊 МОНИТОРИНГ ТОЧНОСТИ И КАЧЕСТВА

### 1. 📈 Ключевые метрики

#### Производительность:

- **Total Return** - общая доходность
- **Monthly Return** - месячная доходность
- **Sharpe Ratio** - коэффициент Шарпа
- **Max Drawdown** - максимальная просадка
- **Win Rate** - процент прибыльных сделок

#### Качество сигналов:

- **Precision** - точность сигналов
- **Recall** - полнота сигналов
- **F1-Score** - гармоническое среднее
- **Confidence Score** - уверенность модели

### 2. 🔍 Система мониторинга

#### Автоматические проверки:

```python
# monitoring/metrics/quality_monitor.py
def check_strategy_quality():
    """Проверка качества стратегии"""

    # Проверка доходности
    if monthly_return < 0.05:  # Менее 5% в месяц
        send_alert("LOW_RETURN", f"Доходность: {monthly_return:.2%}")

    # Проверка Sharpe ratio
    if sharpe_ratio < 1.0:
        send_alert("LOW_SHARPE", f"Sharpe: {sharpe_ratio:.3f}")

    # Проверка просадки
    if max_drawdown > 0.15:  # Более 15%
        send_alert("HIGH_DRAWDOWN", f"Просадка: {max_drawdown:.2%}")

    # Проверка точности ML
    if ml_precision < 0.6:  # Менее 60%
        send_alert("LOW_ML_PRECISION", f"Точность: {ml_precision:.2%}")
```

#### Дашборд мониторинга:

```bash
# Запуск веб-дашборда
python3 monitoring/dashboard/app.py

# Доступ: http://localhost:8080
```

### 3. 📱 Уведомления

#### Настройка алертов:

```python
# config/parameters/alerts_config.py
ALERTS_CONFIG = {
    'email': {
        'enabled': True,
        'recipients': ['trader@example.com'],
        'thresholds': {
            'low_return': 0.05,
            'high_drawdown': 0.15,
            'low_sharpe': 1.0
        }
    },
    'telegram': {
        'enabled': True,
        'bot_token': 'YOUR_BOT_TOKEN',
        'chat_id': 'YOUR_CHAT_ID'
    },
    'slack': {
        'enabled': False,
        'webhook_url': 'YOUR_WEBHOOK_URL'
    }
}
```

---

## 🔄 ДООБУЧЕНИЕ СТРАТЕГИЙ

### 1. 🤖 ML Strategy - Переобучение

#### Автоматическое переобучение:

```python
# strategies/ml/retrain_scheduler.py
def schedule_retraining():
    """Планировщик переобучения"""

    # Ежедневное переобучение
    if datetime.now().hour == 2:  # 2:00 ночи
        retrain_ml_models()

    # Переобучение при снижении качества
    if ml_precision < 0.6:
        retrain_ml_models()

    # Переобучение при изменении рынка
    if market_regime_changed():
        retrain_ml_models()
```

#### Процесс переобучения:

```python
def retrain_ml_models():
    """Переобучение ML моделей"""

    # 1. Сбор новых данных
    new_data = collect_recent_data(days=30)

    # 2. Валидация данных
    if validate_data(new_data):
        # 3. Переобучение моделей
        models = train_models(new_data)

        # 4. Тестирование на валидационной выборке
        validation_score = test_models(models, validation_data)

        # 5. Если качество улучшилось - замена моделей
        if validation_score > current_score:
            replace_models(models)
            log_retraining_success(validation_score)
        else:
            log_retraining_failed(validation_score)
```

### 2. ⚡ Aggressive Strategy - Оптимизация параметров

#### Оптимизация параметров:

```python
# strategies/aggressive/parameter_optimizer.py
def optimize_parameters():
    """Оптимизация параметров агрессивной стратегии"""

    # Используем Optuna для оптимизации
    study = optuna.create_study(direction='maximize')

    def objective(trial):
        # Параметры для оптимизации
        leverage = trial.suggest_float('leverage', 5.0, 50.0)
        profit_target = trial.suggest_float('profit_target', 0.005, 0.05)
        stop_loss = trial.suggest_float('stop_loss', 0.002, 0.02)

        # Тестирование с новыми параметрами
        result = backtest_strategy(leverage, profit_target, stop_loss)

        return result['sharpe_ratio']

    # Запуск оптимизации
    study.optimize(objective, n_trials=100)

    # Применение лучших параметров
    best_params = study.best_params
    update_strategy_parameters(best_params)
```

### 3. 🎯 Combined Strategy - Ребалансировка весов

#### Динамическая ребалансировка:

```python
# strategies/combined/rebalancer.py
def rebalance_strategy_weights():
    """Ребалансировка весов стратегий"""

    # Анализ производительности каждой стратегии
    performance = analyze_strategy_performance()

    # Расчет новых весов на основе производительности
    new_weights = calculate_optimal_weights(performance)

    # Проверка на резкие изменения
    if weight_change_too_large(new_weights):
        # Постепенная корректировка
        new_weights = gradual_adjustment(new_weights)

    # Применение новых весов
    update_strategy_weights(new_weights)

    # Логирование изменений
    log_rebalancing(new_weights, performance)
```

---

## 🛠️ ПРАКТИЧЕСКИЕ КОМАНДЫ

### Запуск системы:

```bash
# 1. Установка зависимостей
pip install -r config/requirements.txt

# 2. Настройка окружения
cp config/environments/.env.example config/environments/.env
# Отредактируйте .env файл

# 3. Запуск мониторинга
python3 monitoring/dashboard/app.py &

# 4. Запуск стратегии
python3 strategies/ml/ml_trading_system.py
```

### Мониторинг:

```bash
# Просмотр логов
tail -f monitoring/logs/*.log

# Проверка метрик
python3 monitoring/metrics/check_all_metrics.py

# Тестирование стратегии
python3 backtesting/engines/quick_test.py
```

### Обслуживание:

```bash
# Переобучение ML моделей
python3 strategies/ml/retrain_scheduler.py

# Оптимизация параметров
python3 strategies/aggressive/parameter_optimizer.py

# Ребалансировка весов
python3 strategies/combined/rebalancer.py
```

---

## ⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ

1. **Всегда тестируйте изменения** на исторических данных
2. **Мониторьте риски** - не превышайте лимиты
3. **Ведите логи** всех операций
4. **Регулярно обновляйте модели** на новых данных
5. **Имейте план отката** на случай проблем

---

## 📞 ПОДДЕРЖКА

При возникновении проблем:

1. Проверьте логи в `monitoring/logs/`
2. Запустите диагностику: `python3 utils/validators/system_check.py`
3. Обратитесь к документации в `docs/`
4. Создайте issue в репозитории проекта

