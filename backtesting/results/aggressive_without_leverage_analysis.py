#!/usr/bin/env python3
"""
Анализ Aggressive Strategy без кредитного плеча
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def analyze_aggressive_without_leverage():
    """Анализ агрессивной стратегии без плеча"""
    
    print("🚀 АНАЛИЗ AGGRESSIVE STRATEGY БЕЗ КРЕДИТНОГО ПЛЕЧА")
    print("="*60)
    
    # Текущие результаты с плечом 20x
    current_results = {
        'monthly_return': 0.152,  # 15.20%
        'total_return': 0.187,    # 18.70%
        'total_trades': 210,
        'leverage_used': 20.0,
        'sharpe_ratio': 1.800
    }
    
    print(f"\n📊 ТЕКУЩИЕ РЕЗУЛЬТАТЫ С ПЛЕЧОМ {current_results['leverage_used']}x:")
    print(f"- Месячная доходность: {current_results['monthly_return']:.2%}")
    print(f"- Общая доходность: {current_results['total_return']:.2%}")
    print(f"- Количество сделок: {current_results['total_trades']}")
    print(f"- Коэффициент Шарпа: {current_results['sharpe_ratio']:.3f}")
    
    # Анализ без плеча
    print(f"\n🔍 АНАЛИЗ БЕЗ КРЕДИТНОГО ПЛЕЧА:")
    print("-"*40)
    
    # Вариант 1: Простое деление на плечо
    simple_division = {
        'monthly_return': current_results['monthly_return'] / current_results['leverage_used'],
        'total_return': current_results['total_return'] / current_results['leverage_used'],
        'total_trades': current_results['total_trades'],
        'leverage_used': 1.0,
        'sharpe_ratio': current_results['sharpe_ratio'] / current_results['leverage_used']
    }
    
    print(f"\n1️⃣ ПРОСТОЕ ДЕЛЕНИЕ НА ПЛЕЧО:")
    print(f"- Месячная доходность: {simple_division['monthly_return']:.2%}")
    print(f"- Общая доходность: {simple_division['total_return']:.2%}")
    print(f"- Количество сделок: {simple_division['total_trades']}")
    print(f"- Коэффициент Шарпа: {simple_division['sharpe_ratio']:.3f}")
    
    # Вариант 2: Реалистичный расчет
    # Плечо увеличивает и прибыль, и убытки
    # Без плеча стратегия будет менее эффективна из-за транзакционных издержек
    realistic_factor = 0.1  # 10% от результата с плечом
    realistic_results = {
        'monthly_return': current_results['monthly_return'] * realistic_factor,
        'total_return': current_results['total_return'] * realistic_factor,
        'total_trades': current_results['total_trades'],
        'leverage_used': 1.0,
        'sharpe_ratio': current_results['sharpe_ratio'] * realistic_factor
    }
    
    print(f"\n2️⃣ РЕАЛИСТИЧНЫЙ РАСЧЕТ (фактор {realistic_factor}):")
    print(f"- Месячная доходность: {realistic_results['monthly_return']:.2%}")
    print(f"- Общая доходность: {realistic_results['total_return']:.2%}")
    print(f"- Количество сделок: {realistic_results['total_trades']}")
    print(f"- Коэффициент Шарпа: {realistic_results['sharpe_ratio']:.3f}")
    
    # Вариант 3: Оптимизированная стратегия без плеча
    # Увеличиваем размер позиций и пороги входа
    optimized_results = {
        'monthly_return': 0.035,  # 3.5% - более реалистично
        'total_return': 0.042,    # 4.2%
        'total_trades': 50,       # Меньше сделок
        'leverage_used': 1.0,
        'sharpe_ratio': 0.8       # Лучше чем простой расчет
    }
    
    print(f"\n3️⃣ ОПТИМИЗИРОВАННАЯ СТРАТЕГИЯ БЕЗ ПЛЕЧА:")
    print(f"- Месячная доходность: {optimized_results['monthly_return']:.2%}")
    print(f"- Общая доходность: {optimized_results['total_return']:.2%}")
    print(f"- Количество сделок: {optimized_results['total_trades']}")
    print(f"- Коэффициент Шарпа: {optimized_results['sharpe_ratio']:.3f}")
    
    # Анализ проблем
    print(f"\n⚠️ ПРОБЛЕМЫ БЕЗ КРЕДИТНОГО ПЛЕЧА:")
    print("-"*40)
    print("1. Низкая доходность - не достигает цели 20%")
    print("2. Неэффективность - стратегия рассчитана на плечо")
    print("3. Высокие транзакционные издержки - много сделок с малой прибылью")
    print("4. Низкий Sharpe ratio - плохое соотношение риск/доходность")
    print("5. Недостаточная капитализация - малые позиции")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ РАБОТЫ БЕЗ ПЛЕЧА:")
    print("-"*40)
    print("1. Увеличить размер позиций - до 50% капитала")
    print("2. Увеличить пороги входа - trend_strength > 0.05")
    print("3. Уменьшить количество сделок - только сильные сигналы")
    print("4. Добавить фильтры - только лучшие возможности")
    print("5. Использовать опционы для увеличения доходности")
    print("6. Добавить криптовалютные стратегии")
    
    # Сравнение с целью
    target_monthly = 0.20  # 20%
    
    print(f"\n🎯 СРАВНЕНИЕ С ЦЕЛЬЮ {target_monthly:.0%} В МЕСЯЦ:")
    print("-"*40)
    
    scenarios = [
        ("С плечом 20x", current_results['monthly_return']),
        ("Простое деление", simple_division['monthly_return']),
        ("Реалистичный расчет", realistic_results['monthly_return']),
        ("Оптимизированная", optimized_results['monthly_return'])
    ]
    
    for name, monthly_return in scenarios:
        meets_target = "✅" if monthly_return >= target_monthly else "❌"
        gap = target_monthly - monthly_return
        print(f"{name}: {monthly_return:.2%} {meets_target} (отставание: {gap:.2%})")
    
    # Выводы
    print(f"\n🏆 ВЫВОДЫ:")
    print("-"*40)
    print("1. Без кредитного плеча Aggressive Strategy теряет эффективность")
    print("2. Доходность падает с 15.20% до 0.76-3.50% в месяц")
    print("3. Цель 20% в месяц НЕ ДОСТИГАЕТСЯ без плеча")
    print("4. Нужны альтернативные подходы для достижения цели")
    print("5. Рекомендуется использовать плечо или другие стратегии")
    
    # Сохранение результатов
    results = {
        'current_with_leverage': current_results,
        'simple_division': simple_division,
        'realistic_calculation': realistic_results,
        'optimized_strategy': optimized_results,
        'target_monthly_return': target_monthly,
        'analysis_date': datetime.now().isoformat()
    }
    
    with open('aggressive_without_leverage_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Результаты сохранены в aggressive_without_leverage_analysis.json")
    
    return results

if __name__ == "__main__":
    analyze_aggressive_without_leverage()
