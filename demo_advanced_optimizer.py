#!/usr/bin/env python3
"""
Демонстрация продвинутого портфельного оптимизатора
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import from advanced optimizer
from advanced_portfolio_optimizer import (
    AdvancedPortfolioOptimizer, RiskLevel, CapitalTier, 
    CommissionStructure, RiskProfiler
)

logger = logging.getLogger(__name__)

def create_demo_strategies() -> Dict:
    """Создать демонстрационные стратегии с разными характеристиками"""
    
    demo_strategies = {
        'Conservative_Bonds': {
            'monthly_return': 0.005,  # 0.5% в месяц
            'volatility': 0.03,
            'max_drawdown': -0.02,
            'sharpe_ratio': 1.2,
            'total_trades': 50,
            'win_rate': 0.75
        },
        'Dividend_Stocks': {
            'monthly_return': 0.008,  # 0.8% в месяц
            'volatility': 0.08,
            'max_drawdown': -0.05,
            'sharpe_ratio': 0.9,
            'total_trades': 30,
            'win_rate': 0.65
        },
        'Balanced_Portfolio': {
            'monthly_return': 0.012,  # 1.2% в месяц
            'volatility': 0.12,
            'max_drawdown': -0.08,
            'sharpe_ratio': 0.8,
            'total_trades': 40,
            'win_rate': 0.60
        },
        'Growth_Stocks': {
            'monthly_return': 0.018,  # 1.8% в месяц
            'volatility': 0.18,
            'max_drawdown': -0.12,
            'sharpe_ratio': 0.7,
            'total_trades': 60,
            'win_rate': 0.58
        },
        'Momentum_Trading': {
            'monthly_return': 0.025,  # 2.5% в месяц
            'volatility': 0.25,
            'max_drawdown': -0.18,
            'sharpe_ratio': 0.6,
            'total_trades': 100,
            'win_rate': 0.55
        },
        'Swing_Trading': {
            'monthly_return': 0.030,  # 3.0% в месяц
            'volatility': 0.30,
            'max_drawdown': -0.22,
            'sharpe_ratio': 0.5,
            'total_trades': 120,
            'win_rate': 0.52
        },
        'High_Frequency': {
            'monthly_return': 0.040,  # 4.0% в месяц
            'volatility': 0.35,
            'max_drawdown': -0.25,
            'sharpe_ratio': 0.4,
            'total_trades': 200,
            'win_rate': 0.50
        },
        'Crypto_Trading': {
            'monthly_return': 0.055,  # 5.5% в месяц
            'volatility': 0.45,
            'max_drawdown': -0.35,
            'sharpe_ratio': 0.35,
            'total_trades': 150,
            'win_rate': 0.48
        },
        'Options_Strategy': {
            'monthly_return': 0.070,  # 7.0% в месяц
            'volatility': 0.50,
            'max_drawdown': -0.40,
            'sharpe_ratio': 0.3,
            'total_trades': 80,
            'win_rate': 0.45
        },
        'Leveraged_ETF': {
            'monthly_return': 0.085,  # 8.5% в месяц
            'volatility': 0.60,
            'max_drawdown': -0.45,
            'sharpe_ratio': 0.25,
            'total_trades': 60,
            'win_rate': 0.42
        },
        'Forex_Scalping': {
            'monthly_return': 0.100,  # 10.0% в месяц
            'volatility': 0.70,
            'max_drawdown': -0.50,
            'sharpe_ratio': 0.2,
            'total_trades': 300,
            'win_rate': 0.40
        },
        'Arbitrage_Bot': {
            'monthly_return': 0.015,  # 1.5% в месяц
            'volatility': 0.05,
            'max_drawdown': -0.03,
            'sharpe_ratio': 1.5,
            'total_trades': 500,
            'win_rate': 0.85
        }
    }
    
    return demo_strategies

def demonstrate_commission_impact():
    """Демонстрация влияния размера капитала на комиссии"""
    
    print("\n" + "="*80)
    print("📊 АНАЛИЗ ВЛИЯНИЯ РАЗМЕРА КАПИТАЛА НА КОМИССИИ")
    print("="*80)
    
    capital_levels = [
        (50_000, "Малый капитал"),
        (250_000, "Начальный капитал"),
        (750_000, "Средний капитал"),
        (2_500_000, "Крупный капитал"),
        (15_000_000, "Институциональный")
    ]
    
    for capital, description in capital_levels:
        commission_structure = CommissionStructure(capital)
        
        # Расчет месячных издержек для типичной торговой активности
        monthly_trades = 50
        avg_trade_size = capital * 0.05  # 5% от капитала на сделку
        
        trade_commission = commission_structure.calculate_trade_cost(avg_trade_size, monthly_trades)
        monthly_fee = commission_structure.get_monthly_fee()
        total_monthly_cost = trade_commission + monthly_fee
        
        cost_percentage = (total_monthly_cost / capital) * 100
        
        print(f"\n{description:20s} ({capital:>10,.0f} ₽):")
        print(f"  Тариф: {commission_structure.tier.value}")
        print(f"  Ставка комиссии: {commission_structure.get_commission_rate():.3%}")
        print(f"  Месячная плата: {monthly_fee:>8,.0f} ₽")
        print(f"  Торговые издержки: {trade_commission:>8,.0f} ₽")
        print(f"  Общие издержки: {total_monthly_cost:>8,.0f} ₽ ({cost_percentage:.3f}%)")

def demonstrate_risk_profiles():
    """Демонстрация профилей риска"""
    
    print("\n" + "="*80)
    print("⚖️ АНАЛИЗ ПРОФИЛЕЙ РИСКА")
    print("="*80)
    
    risk_levels = [
        (RiskLevel.CONSERVATIVE, 1_000_000),
        (RiskLevel.MODERATE, 1_000_000),
        (RiskLevel.AGGRESSIVE, 1_000_000),
        (RiskLevel.SPECULATIVE, 1_000_000)
    ]
    
    risk_names = {
        RiskLevel.CONSERVATIVE: "Консервативный",
        RiskLevel.MODERATE: "Умеренный",
        RiskLevel.AGGRESSIVE: "Агрессивный",
        RiskLevel.SPECULATIVE: "Спекулятивный"
    }
    
    for risk_level, capital in risk_levels:
        profiler = RiskProfiler(risk_level, capital)
        params = profiler.risk_params
        
        print(f"\n{risk_names[risk_level]:15s} профиль:")
        print(f"  Целевая доходность: {params['expected_return_range'][0]:.1%} - {params['expected_return_range'][1]:.1%} в месяц")
        print(f"  Макс. просадка: {params['max_drawdown']:.1%}")
        print(f"  Макс. волатильность: {params['max_volatility']:.1%}")
        print(f"  Макс. позиция: {params['max_single_position']:.1%}")
        print(f"  Мин. диверсификация: {params['min_diversification']} стратегий")
        print(f"  Плечо: до {params['leverage_limit']:.1f}x")

def run_comprehensive_demo():
    """Запустить комплексную демонстрацию"""
    
    print("🚀 ДЕМОНСТРАЦИЯ ПРОДВИНУТОГО ПОРТФЕЛЬНОГО ОПТИМИЗАТОРА")
    print("="*80)
    
    # Показать влияние комиссий
    demonstrate_commission_impact()
    
    # Показать профили риска
    demonstrate_risk_profiles()
    
    # Создать демонстрационные стратегии
    demo_strategies = create_demo_strategies()
    
    print(f"\n📊 СОЗДАННЫЕ ДЕМОНСТРАЦИОННЫЕ СТРАТЕГИИ: {len(demo_strategies)}")
    print("-" * 80)
    
    for name, metrics in demo_strategies.items():
        print(f"{name:20s}: {metrics['monthly_return']:.1%}/мес, "
              f"DD: {metrics['max_drawdown']:.1%}, "
              f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    
    # Тестирование для разных сценариев
    test_scenarios = [
        {
            'capital': 100_000,
            'risk_level': RiskLevel.CONSERVATIVE,
            'target_return': 0.01,  # 1% в месяц
            'description': 'Консервативный инвестор, небольшой капитал'
        },
        {
            'capital': 1_000_000,
            'risk_level': RiskLevel.MODERATE,
            'target_return': 0.03,  # 3% в месяц
            'description': 'Умеренный инвестор, средний капитал'
        },
        {
            'capital': 5_000_000,
            'risk_level': RiskLevel.AGGRESSIVE,
            'target_return': 0.06,  # 6% в месяц
            'description': 'Агрессивный инвестор, крупный капитал'
        },
        {
            'capital': 500_000,
            'risk_level': RiskLevel.SPECULATIVE,
            'target_return': 0.15,  # 15% в месяц
            'description': 'Спекулятивный инвестор, попытка достичь 15% в месяц'
        }
    ]
    
    # Запустить оптимизацию для каждого сценария
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*80}")
        print(f"СЦЕНАРИЙ {i}: {scenario['description']}")
        print(f"{'='*80}")
        
        # Создать оптимизатор
        optimizer = AdvancedPortfolioOptimizer(
            capital_amount=scenario['capital'],
            risk_level=scenario['risk_level'],
            target_return=scenario['target_return']
        )
        
        # Загрузить демонстрационные стратегии
        optimizer.load_strategy_results(demo_strategies)
        
        # Оптимизировать
        results = optimizer.optimize_portfolio_advanced(['risk_adjusted'])
        
        # Показать краткие результаты
        if results.get('success', False) and results['optimization_results'].get('risk_adjusted', {}).get('success', False):
            best_result = results['optimization_results']['risk_adjusted']
            performance = best_result['performance']
            
            print(f"\n✅ ОПТИМИЗАЦИЯ УСПЕШНА:")
            print(f"   Чистая доходность: {performance['monthly_return']:.2%} в месяц")
            print(f"   Целевая доходность: {scenario['target_return']:.2%} в месяц")
            
            target_met = "✅ ДА" if performance['monthly_return'] >= scenario['target_return'] else "❌ НЕТ"
            print(f"   Цель достигнута: {target_met}")
            
            print(f"   Издержки: {performance['total_costs']:.3%} в месяц")
            print(f"   Макс. просадка: {performance['max_drawdown']:.2%}")
            print(f"   Коэффициент Шарпа: {performance['sharpe_ratio']:.3f}")
            
            # Показать топ-3 стратегии в портфеле
            sorted_weights = sorted(best_result['weights'].items(), key=lambda x: x[1], reverse=True)
            print(f"\n   Топ-3 стратегии в портфеле:")
            for strategy, weight in sorted_weights[:3]:
                if weight > 0.01:
                    allocation = scenario['capital'] * weight
                    print(f"     - {strategy}: {weight:.1%} ({allocation:,.0f} ₽)")
            
            # Прогноз роста капитала
            if performance['monthly_return'] > 0:
                print(f"\n   Прогноз роста капитала:")
                for months in [6, 12, 24]:
                    future_value = scenario['capital'] * (1 + performance['monthly_return']) ** months
                    profit = future_value - scenario['capital']
                    print(f"     Через {months:2d} мес.: {future_value:>10,.0f} ₽ (прибыль: {profit:>8,.0f} ₽)")
            
        else:
            print(f"\n❌ ОПТИМИЗАЦИЯ НЕ УДАЛАСЬ")
            print(f"   Причина: {results.get('error', 'Unknown error')}")
            
            # Рекомендации по улучшению
            if scenario['target_return'] > 0.10:  # Если цель больше 10% в месяц
                print(f"\n💡 РЕКОМЕНДАЦИИ:")
                print(f"   - Цель {scenario['target_return']:.1%} в месяц очень амбициозна")
                print(f"   - Рассмотрите снижение до 3-5% в месяц")
                print(f"   - Используйте кредитное плечо (осторожно!)")
                print(f"   - Рассмотрите криптовалюты или деривативы")
    
    # Общие выводы
    print(f"\n{'='*80}")
    print("📋 ОБЩИЕ ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("="*80)
    
    print("""
✅ КЛЮЧЕВЫЕ ПРИНЦИПЫ УСПЕШНОЙ ОПТИМИЗАЦИИ:

1. 💰 РАЗМЕР КАПИТАЛА:
   - Больше капитал = меньше комиссии
   - От 2М ₽ - доступ к лучшим тарифам
   - Институциональные тарифы от 10М ₽

2. ⚖️ УПРАВЛЕНИЕ РИСКАМИ:
   - Консервативный подход: 0.5-2% в месяц, низкие риски
   - Умеренный подход: 2-5% в месяц, средние риски  
   - Агрессивный подход: 5-10% в месяц, высокие риски

3. 🎯 РЕАЛИСТИЧНЫЕ ЦЕЛИ:
   - 20% в месяц требует экстремальных рисков
   - Лучше стремиться к 3-8% в месяц стабильно
   - Реинвестирование увеличивает итоговую прибыль

4. 💼 ДИВЕРСИФИКАЦИЯ:
   - Минимум 4-8 стратегий в портфеле
   - Комбинирование разных стилей торговли
   - Регулярная ребалансировка

⚠️ ВАЖНО ПОМНИТЬ:
- Высокие доходности = высокие риски
- Комиссии существенно влияют на результат
- Тестирование на малых суммах перед масштабированием
- Строгий риск-менеджмент обязателен
""")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_comprehensive_demo()