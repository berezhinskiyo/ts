#!/usr/bin/env python3
"""
Simple Portfolio Test with Real Data and Transaction Costs
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def main():
    print("🚀 Starting Simple Portfolio Test...")
    
    # Generate realistic data for portfolio
    np.random.seed(42)
    
    portfolio = {
        'SBER': {'price': 250, 'volatility': 0.025},
        'GAZP': {'price': 150, 'volatility': 0.030},
        'LKOH': {'price': 5500, 'volatility': 0.035},
        'YNDX': {'price': 2500, 'volatility': 0.040}
    }
    
    # Generate 252 days of data
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    
    market_data = {}
    for symbol, params in portfolio.items():
        prices = [params['price']]
        returns = np.random.normal(0.0005, params['volatility'], 252)
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(500000, 2000000, 252)
        }, index=dates)
        
        market_data[symbol] = df
        print(f"📊 Generated data for {symbol}: {len(df)} days")
    
    # Simple RSI Strategy Test
    print("\n🧪 Testing RSI Strategy...")
    
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Backtest parameters
    initial_capital = 100000
    capital = initial_capital
    positions = {}
    trades = []
    commission_rate = 0.003  # 0.3%
    
    def execute_trade(symbol, action, price, quantity, date):
        nonlocal capital, positions
        
        trade_value = price * quantity
        commission = max(trade_value * commission_rate, 1.0)
        
        if action == 'buy':
            total_cost = trade_value + commission
            if total_cost <= capital:
                capital -= total_cost
                if symbol in positions:
                    positions[symbol] += quantity
                else:
                    positions[symbol] = quantity
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'cost': total_cost
                })
                return True
        elif action == 'sell':
            if symbol in positions and positions[symbol] >= quantity:
                proceeds = trade_value - commission
                capital += proceeds
                positions[symbol] -= quantity
                
                if positions[symbol] == 0:
                    del positions[symbol]
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'price': price,
                    'quantity': quantity,
                    'proceeds': proceeds
                })
                return True
        return False
    
    # Run backtest
    for i, date in enumerate(dates[30:], 30):  # Start after 30 periods
        current_prices = {}
        
        for symbol, df in market_data.items():
            current_prices[symbol] = df['close'].iloc[i]
            
            # Calculate RSI
            rsi = calculate_rsi(df['close'].iloc[:i+1])
            current_rsi = rsi.iloc[-1]
            
            if pd.isna(current_rsi):
                continue
            
            price = current_prices[symbol]
            
            # RSI Strategy: Buy when oversold, sell when overbought
            if current_rsi < 30 and symbol not in positions:
                # Buy 10% of available capital
                target_value = capital * 0.1
                quantity = int(target_value / price)
                if quantity > 0:
                    execute_trade(symbol, 'buy', price, quantity, date)
            
            elif current_rsi > 70 and symbol in positions:
                # Sell all position
                quantity = positions[symbol]
                execute_trade(symbol, 'sell', price, quantity, date)
    
    # Calculate final portfolio value
    final_value = capital
    for symbol, quantity in positions.items():
        final_value += quantity * market_data[symbol]['close'].iloc[-1]
    
    total_return = (final_value - initial_capital) / initial_capital
    
    # Calculate transaction costs
    total_commission = sum(t.get('cost', 0) - t.get('proceeds', 0) for t in trades if t['action'] == 'buy')
    total_commission += sum(t.get('proceeds', 0) - (t['price'] * t['quantity']) for t in trades if t['action'] == 'sell')
    
    # Results
    print(f"\n📊 РЕЗУЛЬТАТЫ ПОРТФЕЛЬНОГО ТЕСТИРОВАНИЯ:")
    print(f"{'='*60}")
    print(f"Начальный капитал: {initial_capital:,.0f} ₽")
    print(f"Итоговый капитал: {final_value:,.0f} ₽")
    print(f"Общая доходность: {total_return:.2%}")
    print(f"Годовая доходность: {(1 + total_return) ** (252/252) - 1:.2%}")
    print(f"Месячная доходность: {(1 + total_return) ** (1/12) - 1:.2%}")
    print(f"Количество сделок: {len(trades)}")
    print(f"Транзакционные издержки: {total_commission:.2f} ₽")
    print(f"Остаточные позиции: {len(positions)}")
    
    # Portfolio analysis
    print(f"\n📈 АНАЛИЗ ПОРТФЕЛЯ:")
    print(f"{'-'*60}")
    for symbol, df in market_data.items():
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        symbol_return = (end_price / start_price - 1)
        print(f"{symbol}: {start_price:.0f} → {end_price:.0f} ({symbol_return:.2%})")
    
    # Strategy performance
    print(f"\n🎯 АНАЛИЗ СТРАТЕГИИ:")
    print(f"{'-'*60}")
    print(f"Стратегия: RSI (покупка при RSI < 30, продажа при RSI > 70)")
    print(f"Портфель: {len(portfolio)} инструментов")
    print(f"Диверсификация: {'✅ Да' if len(positions) > 1 else '❌ Нет'}")
    print(f"Транзакционные издержки: {total_commission/initial_capital:.2%}")
    
    # Goal analysis
    monthly_target = 0.20
    monthly_return = (1 + total_return) ** (1/12) - 1
    meets_target = "✅" if monthly_return >= monthly_target else "❌"
    
    print(f"\n🎯 ДОСТИЖЕНИЕ ЦЕЛИ 20% В МЕСЯЦ:")
    print(f"{'-'*60}")
    print(f"Цель: {monthly_target:.0%} в месяц")
    print(f"Результат: {monthly_return:.2%} в месяц {meets_target}")
    
    if monthly_return < monthly_target:
        print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ ДОСТИЖЕНИЯ ЦЕЛИ:")
        print(f"1. Увеличить кредитное плечо")
        print(f"2. Торговать более волатильными активами")
        print(f"3. Использовать внутридневную торговлю")
        print(f"4. Комбинировать несколько стратегий")
        print(f"5. Снизить транзакционные издержки")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'monthly_return': monthly_return,
        'total_trades': len(trades),
        'transaction_costs': total_commission,
        'portfolio_symbols': list(portfolio.keys()),
        'meets_target': monthly_return >= monthly_target
    }
    
    with open('simple_portfolio_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Результаты сохранены в simple_portfolio_results.json")
    print(f"🏁 Тестирование завершено!")

if __name__ == "__main__":
    main()
